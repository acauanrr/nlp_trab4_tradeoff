# -*- coding: utf-8 -*-
"""
Fine-tuning de LLMs com LoRA + 4-bit quant (TRL ≥ 0.19).

Executar:
    python scripts/train_lora.py \
        --config_file configs/lora_config_2.json \
        --dataset_path data/spider/train_formatted.jsonl
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

import numpy as np, torch, transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# --------------------------------------------------------------------------- #
#  Seed para reprodutibilidade                                                #
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


# --------------------------------------------------------------------------- #
#  Pipeline principal                                                         #
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    # 1) Hiperparâmetros
    cfg = json.load(open(args.config_file, "r", encoding="utf-8"))
    set_seed(args.seed)
    model_id: str = cfg["model_id"]

    # 2) Tokenizer + modelo em 4-bit
    print(f"Carregando modelo base: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 3) LoRA
    print("Configurando LoRA…")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 4) Dataset
    print(f"Carregando dataset de: {args.dataset_path}")
    train_ds = load_dataset("json", data_files=args.dataset_path, split="train")

    # 5) Config de treinamento (substitui TrainingArguments)
    output_dir = Path(args.output_base_dir) / Path(args.config_file).stem
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=True,
        seed=args.seed,
        logging_steps=25,
        save_steps=200,
        max_steps=400,                 # limite rápido
        # Se a coluna de texto NÃO se chamar "text", declare aqui:
        # dataset_text_field="prompt",
    )

    # 6) Trainer (sem tokenizer=…)
    trainer = SFTTrainer(
        model,                         # pode ser str ou objeto
        train_dataset=train_ds,
        args=sft_cfg,
        peft_config=lora_cfg,
    )

    # 7) Treino
    print("Iniciando o fine-tuning…")
    trainer.train()

    # 8) Salvar adaptador
    final_adapter = Path(output_dir) / "final_adapter"
    final_adapter.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_adapter)
    print(f"Adaptador LoRA salvo em: {final_adapter}")


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser("Fine-tuning LLM + LoRA (Spider dataset)")
    p.add_argument("--config_file", required=True,
                   help="JSON com hiperparâmetros do LoRA")
    p.add_argument("--dataset_path", default="data/spider/train_formatted.jsonl",
                   help="Caminho do dataset (.jsonl)")
    p.add_argument("--output_base_dir", default="results",
                   help="Diretório para salvar checkpoints")
    p.add_argument("--seed", type=int, default=42, help="Seed global")
    main(p.parse_args())
