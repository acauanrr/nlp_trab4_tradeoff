# project_root/scripts/eval_mmlu.py

import argparse
import json
import random
import re
from collections import defaultdict

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Funções de Setup e Geração ---

def set_seed(seed: int):
    """Fixar seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_model_for_eval(model_id, lora_adapter_path=None):
    """Carrega o modelo base e, opcionalmente, aplica um adaptador LoRA."""
    print(f"Carregando modelo base: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if lora_adapter_path:
        print(f"Aplicando adaptador LoRA de: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def format_mmlu_example(example: dict, include_answer=True):
    """Formata um único exemplo MMLU em texto."""
    choices = [f"{chr(65+i)}) {choice}" for i, choice in enumerate(example['choices'])]
    prompt = f"Question: {example['question']}\nChoices:\n" + "\n".join(choices)
    if include_answer:
        answer_letter = chr(65 + example['answer'])
        prompt += f"\nAnswer: {answer_letter}"
    else:
        prompt += f"\nAnswer:"
    return prompt

def create_4_shot_prompt(test_example: dict, shot_pool: dict) -> str:
    """Cria um prompt 4-shot usando exemplos da mesma categoria."""
    # Mapeia o domínio para a subcategoria original para buscar os shots
    subject_map = {
        "STEM": "high_school_computer_science",
        "Humanidades": "philosophy",
        "Sociais": "econometrics"
    }
    subject = subject_map[test_example['domain']]
    
    # Seleciona 4 exemplos aleatórios do pool da mesma categoria
    shots = random.sample(shot_pool[subject], 4)
    
    # Formata o prompt
    prompt_prefix = "The following are multiple choice questions (with answers).\n\n"
    shot_prompts = [format_mmlu_example(shot) for shot in shots]
    test_prompt = format_mmlu_example(test_example, include_answer=False)
    
    return prompt_prefix + "\n\n".join(shot_prompts) + "\n\n" + test_prompt

def parse_model_output(raw_output: str) -> str:
    """Extrai a letra da resposta (A, B, C, ou D) da saída do modelo."""
    # Procura por "Answer: A" ou apenas "A" no início da string
    match = re.search(r"^\s*([A-D])", raw_output.strip())
    if match:
        return match.group(1)
    return "Z" # Retorna uma resposta inválida se não encontrar


# --- Função Principal de Avaliação ---

def main(args):
    set_seed(args.seed)
    
    # 1. Carregar Modelo e Tokenizer
    model, tokenizer = load_model_for_eval(args.model_id, args.lora_adapter_path)

    # 2. Carregar Datasets
    print(f"Carregando dataset de teste de: {args.test_data_path}")
    test_dataset = load_dataset('json', data_files=args.test_data_path, split='train')
    
    print("Carregando dataset para os exemplos 4-shot (cais/mmlu dev split)...")
    shot_subjects = ["high_school_computer_science", "philosophy", "econometrics"]
    shot_pool = defaultdict(list)
    for subject in shot_subjects:
        # Usamos o split 'dev' para os shots, conforme boas práticas
        ds = load_dataset("cais/mmlu", subject, split="dev")
        for item in ds:
            shot_pool[subject].append(item)

    # 3. Executar Avaliação
    results = {
        "overall": {"correct": 0, "total": 0},
        "STEM": {"correct": 0, "total": 0},
        "Humanidades": {"correct": 0, "total": 0},
        "Sociais": {"correct": 0, "total": 0}
    }

    for item in tqdm(test_dataset, desc="Avaliando MMLU 4-shot"):
        # a. Construir o prompt
        prompt_text = create_4_shot_prompt(item, shot_pool)
        
        # b. Formatar com o template de chat do Llama-3
        chat = [{"role": "user", "content": prompt_text}]
        prompt_templated = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt_templated, return_tensors="pt").to("cuda")

        # c. Gerar a resposta
        outputs = model.generate(**inputs, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id)
        raw_response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # d. Parse e comparação
        predicted_answer = parse_model_output(raw_response)
        correct_answer = chr(65 + item['answer'])
        domain = item['domain']

        # e. Registrar resultados
        results[domain]['total'] += 1
        results['overall']['total'] += 1
        if predicted_answer == correct_answer:
            results[domain]['correct'] += 1
            results['overall']['correct'] += 1

    # 4. Calcular e Apresentar Acurácias
    print("\n--- Resultados da Avaliação MMLU ---")
    mode_str = "Fine-Tuned" if args.lora_adapter_path else "Baseline"
    print(f"Modelo: {args.model_id} ({mode_str})")
    print("-" * 40)
    
    for category, scores in results.items():
        if scores['total'] > 0:
            accuracy = (scores['correct'] / scores['total']) * 100
            print(f"Categoria: {category.capitalize():<12} | Acurácia: {accuracy:.2f}% ({scores['correct']}/{scores['total']})")
    
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia um modelo LLM no subset MMLU com 4-shot.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3-8B-Instruct", help="ID do modelo base no Hugging Face.")
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Caminho para o adaptador LoRA treinado (opcional).")
    parser.add_argument("--test_data_path", type=str, default="data/mmlu_subset/mmlu_150_eval.jsonl", help="Caminho para o arquivo de teste MMLU formatado.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    
    args = parser.parse_args()
    main(args)