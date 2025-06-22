# project_root/scripts/eval_spider.py  ─── VERSÃO OTIMIZADA (2025-06-22)

import sys, os, argparse, json, random, re, gc
import numpy as np
from tqdm import tqdm
import torch

# ────────────────────  DeepEval  ────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from custom_metrics.execution_accuracy import ExecutionAccuracy

# ─────────────────── Transformers / PEFT  ───────────
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from typing import List, Dict

# ────────────────────  Constantes  ──────────────────
MODEL_ID_CORRETO = "meta-llama/Meta-Llama-3-8B-Instruct"

# ────────────────────  CLI  ─────────────────────────
parser = argparse.ArgumentParser(
    description="Avalia um modelo LLM no dataset Spider."
)
parser.add_argument("--mode", required=True, choices=["baseline", "finetuned"])
parser.add_argument("--model_id", default=MODEL_ID_CORRETO)
parser.add_argument("--lora_adapter_path")
parser.add_argument("--dev_path",   default="data/spider/dev.json")
parser.add_argument("--train_path", default="data/spider/train.json")
parser.add_argument("--tables_path",default="data/spider/tables.json")
parser.add_argument("--output_file", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_samples", type=int)
parser.add_argument("--batch_size",  type=int, default=4, help="↓ VRAM")  # DEFAULT ↓
args = parser.parse_args()

# ───────────────────── Helpers ──────────────────────
GEN_KWARGS = dict(max_new_tokens=160, use_cache=False, do_sample=False)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_model_for_eval(model_id: str, lora_adapter_path=None):
    print(f"Carregando modelo: {model_id}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # ↓ memória em fp16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_adapter_path:
        print(f"→ Aplicando LoRA: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok

# ---------- prompt helpers (unchanged) ----------
def create_few_shot_prompt(question:str, schema:str, examples:List[Dict])->str:
    blocks = "\n\n".join(
        f"### SCHEMA\n{ex['schema']}\n\n### QUESTION\n{ex['question']}\n\n### SQL\n{ex['query']}"
        for ex in examples
    )
    return (
        "### INSTRUCTION\n"
        "Given the database schema below, write a SQL query that answers the following question.\n\n"
        f"{blocks}\n\n### SCHEMA\n{schema}\n\n### QUESTION\n{question}\n\n### SQL\n"
    )

def get_schema(db_id:str, tables:dict)->str:
    if db_id not in tables: return ""
    db = tables[db_id]; out=[]
    for t_idx, t_name in enumerate(db["table_names_original"]):
        cols=[
            f"  {c[1]} {db['column_types'][i]}"
            for i,c in enumerate(db["column_names_original"]) if c[0]==t_idx
        ]
        pk=[c[1] for i,c in enumerate(db["column_names_original"])
            if i in db["primary_keys"] and c[0]==t_idx]
        if pk: cols.append(f"  PRIMARY KEY ({', '.join(pk)})")
        out.append(f"CREATE TABLE {t_name} (\n"+",\n".join(cols)+"\n);")
    return "\n".join(out)

def extract_sql_from_response(resp:str)->str:
    m=re.search(r"```sql\s*(.*?)\s*```", resp, re.S)
    if m: return m.group(1).strip()
    cleaned=re.sub(r"^(?:Here(?:'s| is)|The SQL query is:|### SQL|```sql|```)\s*",
                   "", resp.strip(), flags=re.I)
    return cleaned.split(";")[0].strip() or "-- ERROR: Empty query"

# ─────────────────── Main workflow ──────────────────
def create_and_run_evaluation():
    set_seed(args.seed)
    model, tok = load_model_for_eval(args.model_id, args.lora_adapter_path)
    model.eval()

    dev = json.load(open(args.dev_path))
    if args.max_samples:
        print(f"Avaliando em amostra de {args.max_samples}")
        dev=dev[:args.max_samples]

    tables={db["db_id"]:db for db in json.load(open(args.tables_path))}

    # few-shot
    few=[]
    if args.mode=="baseline":
        train=json.load(open(args.train_path))
        for ex in random.sample(train, k=min(3,len(train))):
            few.append({"question":ex["question"],
                        "schema":get_schema(ex["db_id"], tables),
                        "query":ex["query"]})

    prompts, metas=[],[]
    print("1. Preparando prompts …")
    for item in tqdm(dev):
        schema=get_schema(item["db_id"], tables)
        if not schema: continue
        q=item["question"]
        prompt=create_few_shot_prompt(q,schema,few) if args.mode=="baseline" \
               else f"### INSTRUCTION\nGiven the database schema below, write a SQL query that answers the following question.\n\n### SCHEMA\n{schema}\n\n### QUESTION\n{q}\n\n### SQL\n"
        chat=[{"role":"user","content":prompt}]
        prompts.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        metas.append(item)

    cases = []
    raw   = []
    bs=args.batch_size
    print(f"\n2. Gerando SQL em lotes de {bs} …")
    with torch.no_grad():
        for i in tqdm(range(0,len(prompts),bs)):
            batch_prompts = prompts[i:i+bs]
            meta_batch    = metas[i:i+bs]

            inputs=tok(batch_prompts, return_tensors="pt",
                       padding=True, truncation=True).to("cuda")
            outs=model.generate(**inputs, **GEN_KWARGS)

            decoded=tok.batch_decode(outs, skip_special_tokens=True)
            for j,res in enumerate(decoded):
                prompt_len=len(tok.decode(inputs.input_ids[j], skip_special_tokens=True))
                sql=extract_sql_from_response(res[prompt_len:])
                item=meta_batch[j]
                cases.append(
                    LLMTestCase(input=batch_prompts[j], actual_output=sql,
                                expected_output=item["query"], context=[item["db_id"]])
                )
                raw.append({"db_id":item["db_id"],"question":item["question"],
                            "predicted_sql":sql,"ground_truth_sql":item["query"]})
            # Libera VRAM e reduz fragmentação
            del inputs, outs; torch.cuda.empty_cache(); gc.collect()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    json.dump(raw, open(args.output_file,"w"), indent=4, ensure_ascii=False)
    print(f"\nOutputs salvos em: {args.output_file}")

    print("\n"+"="*20+" AVALIANDO COM DEEPEVAL "+"="*20)
    metric=ExecutionAccuracy()
    evaluate(cases,[metric])
    print("✓ Avaliação concluída!  Execute `deepeval view` para inspeção.")

# ────────────────────────────────────────────────────
if __name__=="__main__":
    if args.mode=="finetuned" and not args.lora_adapter_path:
        parser.error("--lora_adapter_path é obrigatório no modo 'finetuned'.")
    create_and_run_evaluation()
