# project_root/scripts/eval_spider.py

import sys
import os
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import re

# Adiciona o diretório raiz ao path para encontrar a métrica customizada
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Módulos do DeepEval
import deepeval
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from custom_metrics.execution_accuracy import ExecutionAccuracy

# Módulos do Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict

# --- Constantes Globais ---
MODEL_ID_CORRETO = "meta-llama/Meta-Llama-3-8B-Instruct"
DB_ROOT_PATH = "data/spider/database"

# --- Análise dos Argumentos de Linha de Comando ---
parser = argparse.ArgumentParser(description="Avalia um modelo LLM no dataset Spider.")
parser.add_argument("--mode", type=str, required=True, choices=['baseline', 'finetuned'], help="Modo de avaliação: baseline ou com fine-tuning.")
parser.add_argument("--model_id", type=str, default=MODEL_ID_CORRETO, help="ID do modelo base no Hugging Face.")
parser.add_argument("--lora_adapter_path", type=str, default=None, help="Caminho para o adaptador LoRA treinado (para o modo 'finetuned').")
parser.add_argument("--dev_path", type=str, default="data/spider/dev.json", help="Caminho para o arquivo de desenvolvimento do Spider.")
parser.add_argument("--train_path", type=str, default="data/spider/train.json", help="Caminho para o arquivo de treino do Spider (usado para few-shot).")
parser.add_argument("--tables_path", type=str, default="data/spider/tables.json", help="Caminho para o arquivo de schemas do Spider.")
parser.add_argument("--output_file", type=str, required=True, help="Arquivo para salvar as queries geradas.")
parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
parser.add_argument("--max_samples", type=int, default=None, help="Número máximo de exemplos a serem avaliados para um teste rápido.")
args = parser.parse_args()


# --- Funções de Setup e Geração ---

def set_seed(seed: int):
    """Fixa as seeds para garantir resultados consistentes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_model_for_eval(model_id, lora_adapter_path=None):
    """Carrega o modelo (e o adaptador LoRA, se aplicável) para avaliação."""
    print(f"Carregando modelo: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    if lora_adapter_path:
        print(f"Aplicando adaptador LoRA de: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def create_few_shot_prompt(question: str, schema: str, examples: List[Dict]) -> str:
    """Constrói um prompt few-shot com exemplos de treino."""
    example_str = ""
    for ex in examples:
        example_str += f"### SCHEMA\n{ex['schema']}\n\n### QUESTION\n{ex['question']}\n\n### SQL\n{ex['query']}\n\n---\n\n"
    full_prompt = f"### INSTRUCTION\nGiven the database schema below, write a SQL query that answers the following question.\n\n{example_str}### SCHEMA\n{schema}\n\n### QUESTION\n{question}\n\n### SQL\n"
    return full_prompt

def get_schema(db_id: str, tables_data: dict) -> str:
    """Extrai o schema 'CREATE TABLE' para um banco de dados específico."""
    if db_id not in tables_data: return ""
    schema_str = ""
    for table in tables_data[db_id]['table_names_original']:
        table_idx = tables_data[db_id]['table_names_original'].index(table)
        cols = []
        for col_idx, (col_name_original, col_type) in enumerate(zip(tables_data[db_id]['column_names_original'], tables_data[db_id]['column_types'])):
            if col_name_original[0] == table_idx:
                 cols.append(f"  {col_name_original[1]} {col_type}")
        pk_cols = [col[1] for i, col in enumerate(tables_data[db_id]['column_names_original']) if i in tables_data[db_id]['primary_keys'] and col[0] == table_idx]
        if pk_cols: cols.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")
        schema_str += f"CREATE TABLE {table} (\n" + ",\n".join(cols) + "\n);\n"
    return schema_str.strip()

def generate_and_extract_sql(model, tokenizer, prompt_text: str) -> str:
    """Gera a resposta completa e extrai apenas a query SQL."""
    chat = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # Lógica de extração do SQL para limpar a saída do modelo
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
    if sql_match:
        sql_query = sql_match.group(1).strip()
    else:
        cleaned_response = re.sub(r"^(Here's|Here is|The SQL query is:|To answer the question, you can use the following SQL query:|### SQL|### ANSWER|```sql|```)\s*", "", response.strip(), flags=re.IGNORECASE)
        sql_query = cleaned_response.split(';')[0].strip()

    return sql_query if sql_query else "-- ERROR: Empty query extracted"

def create_and_run_evaluation():
    """Função principal que orquestra a geração e a avaliação."""
    
    set_seed(args.seed)
    
    model, tokenizer = load_model_for_eval(args.model_id, args.lora_adapter_path)
    
    dev_data = json.load(open(args.dev_path))
    if args.max_samples is not None:
        print(f"Avaliando em uma amostra de {args.max_samples} exemplos.")
        dev_data = dev_data[:args.max_samples]

    with open(args.tables_path, 'r') as f:
        tables_raw = json.load(f)
    tables_data = {db['db_id']: db for db in tables_raw}

    few_shot_examples = []
    if args.mode == 'baseline':
        train_data = json.load(open(args.train_path))
        num_samples = min(3, len(train_data))
        sample_indices = random.sample(range(len(train_data)), num_samples)
        for i in sample_indices:
            ex = train_data[i]
            ex_schema = get_schema(ex['db_id'], tables_data)
            few_shot_examples.append({"question": ex['question'], "schema": ex_schema, "query": ex['query']})

    eval_cases = []
    all_outputs = []
    for item in tqdm(dev_data, desc="Gerando queries SQL"):
        db_id = item['db_id']
        schema = get_schema(db_id, tables_data)
        if not schema: continue
        
        question = item['question']
        expected_output = item['query']
        
        if args.mode == 'baseline':
            prompt_text = create_few_shot_prompt(question, schema, few_shot_examples)
        else:
            prompt_text = f"### INSTRUCTION\nGiven the database schema below, write a SQL query that answers the following question.\n\n### SCHEMA\n{schema}\n\n### QUESTION\n{question}\n\n### SQL\n"

        actual_output = generate_and_extract_sql(model, tokenizer, prompt_text)
        
        test_case = LLMTestCase(
            input=prompt_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=[db_id]
        )
        eval_cases.append(test_case)
        all_outputs.append({
            'db_id': db_id,
            'question': question,
            'predicted_sql': actual_output,
            'ground_truth_sql': expected_output
        })

    # Salva as predições brutas (já limpas)
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_outputs, f, indent=4, ensure_ascii=False)
    print(f"Outputs brutos salvos em: {args.output_file}")

    # Executa a avaliação com DeepEval
    print("\n" + "="*20 + " INICIANDO AVALIAÇÃO COM DEEPEVAL " + "="*20)
    metric = ExecutionAccuracy()
    results = evaluate(eval_cases, [metric])
    print("Avaliação concluída!")
    print("\nPara visualizar os resultados detalhados, execute: deepeval view")


if __name__ == "__main__":
    if args.mode == 'finetuned' and not args.lora_adapter_path:
        parser.error("--lora_adapter_path é obrigatório no modo 'finetuned'.")

    create_and_run_evaluation()
