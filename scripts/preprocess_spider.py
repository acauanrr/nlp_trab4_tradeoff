# project_root/scripts/preprocess_spider.py

import json
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm
import argparse

def get_schema(db_id: str, tables_data: dict) -> str:
    """
    Extrai o schema 'CREATE TABLE' para um determinado db_id.
    """
    if db_id not in tables_data:
        return ""
    
    schema_str = ""
    for table in tables_data[db_id]['table_names_original']:
        table_idx = tables_data[db_id]['table_names_original'].index(table)
        
        # Colunas e tipos
        cols = []
        for col_idx, (col_name, col_type) in enumerate(zip(tables_data[db_id]['column_names_original'], tables_data[db_id]['column_types'])):
            if col_name[0] == table_idx:
                cols.append(f"  {col_name[1]} {col_type}")

        # Chaves primárias e estrangeiras
        pk_cols = [col[1] for i, col in enumerate(tables_data[db_id]['column_names_original']) if i in tables_data[db_id]['primary_keys'] and col[0] == table_idx]
        if pk_cols:
            cols.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")

        schema_str += f"CREATE TABLE {table} (\n" + ",\n".join(cols) + "\n);\n"

    # Adicionando Foreign Keys
    for fk in tables_data[db_id]['foreign_keys']:
        col_id, other_col_id = fk
        col_table_idx, col_name = tables_data[db_id]['column_names_original'][col_id]
        other_table_idx, other_col_name = tables_data[db_id]['column_names_original'][other_col_id]
        col_table_name = tables_data[db_id]['table_names_original'][col_table_idx]
        other_table_name = tables_data[db_id]['table_names_original'][other_table_idx]
        # Esta parte é simplificada; para uma implementação robusta, o ALTER TABLE seria adicionado fora do CREATE.
        # Mas para fins de contexto do LLM, isso é suficiente.
        schema_str += f"-- ALTER TABLE {col_table_name} ADD FOREIGN KEY ({col_name}) REFERENCES {other_table_name}({other_col_name});\n"
        
    return schema_str.strip()

def create_instruction(question: str, schema: str) -> str:
    """
    Cria a instrução formatada para o modelo.
    """
    return f"""### INSTRUCTION\nGiven the database schema below, write a SQL query that answers the following question.

### SCHEMA
{schema}

### QUESTION
{question}

### SQL
"""

def format_chat_template(instruction: str, sql: str) -> dict:
    """
    Formata o exemplo no template de chat do Llama-3-Instruct.
    """
    # Usando o template oficial do Llama-3 <|start_header_id|>...<|eot_id|>
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
    completion = f"<|start_header_id|>assistant<|end_header_id|>\n\n{sql}<|eot_id|>"
    return {"text": prompt + completion}


def main(args):
    print("Carregando datasets do Spider...")
    # O arquivo 'train_spider.json' foi upado pelo usuário
    train_data = json.load(open(args.train_path))
    dev_data = json.load(open(args.dev_path))
    
    # Carregando informações de schema
    with open(args.tables_path, 'r') as f:
        tables_raw = json.load(f)
    tables_data = {db['db_id']: db for db in tables_raw}

    print("Processando e formatando os dados...")
    
    processed_datasets = {}
    for split_name, split_data in [('train', train_data), ('dev', dev_data)]:
        formatted_data = []
        for item in tqdm(split_data, desc=f"Processando {split_name}"):
            db_id = item['db_id']
            schema = get_schema(db_id, tables_data)
            if not schema:
                continue
            
            instruction = create_instruction(item['question'], schema)
            formatted_entry = format_chat_template(instruction, item['query'])
            formatted_data.append(formatted_entry)
        
        # Cria o diretório de saída se não existir
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{split_name}_formatted.jsonl")
        
        with open(output_path, 'w') as f:
            for entry in formatted_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Dados formatados salvos em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processa o dataset Spider para fine-tuning.")
    parser.add_argument("--train_path", type=str, default="data/spider/train.json", help="Caminho para o train split do Spider.")
    parser.add_argument("--dev_path", type=str, default="data/spider/dev.json", help="Caminho para o dev split do Spider.")
    parser.add_argument("--tables_path", type=str, default="data/spider/tables.json", help="Caminho para o arquivo de schemas (tables.json).")
    parser.add_argument("--output_dir", type=str, default="data/spider", help="Diretório para salvar os arquivos formatados.")
    
    args = parser.parse_args()
    main(args)