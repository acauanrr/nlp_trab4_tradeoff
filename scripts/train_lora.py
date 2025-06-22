# project_root/scripts/train_lora.py

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# --- ALTERAÇÃO IMPORTANTE: Importando SFTConfig ---
from trl import SFTTrainer, SFTConfig
import argparse
import json
import os
import random
import numpy as np

def set_seed(seed: int):
    """Fixar seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

def main(args):
    # --- 1. Setup e Configuração ---
    set_seed(args.seed)

    # Carregar config de hiperparâmetros
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    model_id = config['model_id']
    
    # --- 2. Carregar Modelo e Tokenizer ---
    print(f"Carregando modelo base: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # --- 3. Configuração do LoRA (PEFT) ---
    print("Configurando LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Carregar Dataset ---
    print(f"Carregando dataset de: {args.dataset_path}")
    train_dataset = load_dataset('json', data_files=args.dataset_path, split='train')

    # --- 5. Configurar o Trainer usando SFTConfig ---
    output_dir = os.path.join(args.output_base_dir, os.path.splitext(os.path.basename(args.config_file))[0])
    
    # --- SOLUÇÃO DEFINITIVA: Usar SFTConfig em vez de TrainingArguments ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=200,             # Salva um checkpoint a cada 200 passos
        logging_steps=25,
        fp16=True,
        seed=args.seed,
        report_to="none",
        max_steps=400,              # Otimização para agilizar o treinamento
        
        # Parâmetros específicos do SFT agora são passados aqui:
        dataset_text_field="text",
        max_seq_length=1024,
    )

    # --- CORREÇÃO FINAL: Chamada ao SFTTrainer simplificada ---
    # Removidos os argumentos que causavam erro, pois eles são lidos do SFTConfig e do modelo.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    # --- 6. Treinar o Modelo ---
    print("Iniciando o fine-tuning...")
    trainer.train()

    # --- 7. Salvar o Adaptador LoRA ---
    final_adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    print(f"Adaptador LoRA salvo em: {final_adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune um modelo LLM com LoRA no dataset Spider.")
    parser.add_argument("--config_file", type=str, required=True, help="Caminho para o arquivo de configuração JSON dos hiperparâmetros.")
    parser.add_argument("--dataset_path", type=str, default="data/spider/train_formatted.jsonl", help="Caminho para o dataset de treino formatado.")
    parser.add_argument("--output_base_dir", type=str, default="results", help="Diretório base para salvar os resultados do treino.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    
    args = parser.parse_args()
    main(args)