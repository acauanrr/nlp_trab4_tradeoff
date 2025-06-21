# Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs

Este projeto implementa um pipeline completo para o **fine-tuning** do modelo `meta-llama/Meta-Llama-3-8B-Instruct` na tarefa de **Text-to-SQL**, utilizando o dataset **Spider**. O objetivo é quantificar o ganho de performance na tarefa especializada e, ao mesmo tempo, medir a regressão em tarefas de conhecimento geral (MMLU), fornecendo uma análise crítica do *trade-off* entre **especialização e generalização** em LLMs.

---

## 1. Estrutura do Projeto

```
/
├── configs/
│   ├── lora_config_1.json
│   └── lora_config_2.json
├── custom_metrics/
│   └── execution_accuracy.py
├── data/
│   ├── mmlu_subset/
│   │   └── mmlu_150_eval.jsonl
│   └── spider/
│       ├── database/
│       ├── dev.json
│       ├── tables.json
│       └── train.json
├── results/
│   └── (outputs dos modelos serão salvos aqui)
├── scripts/
│   ├── preprocess_spider.py
│   ├── train_lora.py
│   ├── eval_spider.py
│   └── eval_mmlu.py
├── requirements.txt
└── README.md
```

---

## 2. Setup do Ambiente

### Pré-requisitos:
- Python 3.9+
- NVIDIA GPU com suporte a CUDA (recomendado VRAM ≥ 12GB para QLoRA)
- `git` e `git-lfs` instalados

### Passos:

1. Clone o repositório:
```bash
git clone [https://github.com/acauanrr/nlp_trab4_tradeoff.git](https://github.com/acauanrr/nlp_trab4_tradeoff.git)
cd nlp_trab4_tradeoff
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv

# Windows
.\.venv\Scriptsctivate

# Linux/macOS
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

> **Nota**: Se estiver no Windows e encontrar erros com `bitsandbytes`, consulte soluções com versões pré-compiladas. Os scripts já estão adaptados para contornar o problema, se necessário.

4. Autentique-se no Hugging Face:
```bash
huggingface-cli login
```
> Você precisará de um token com acesso ao modelo Llama-3.

5. Baixe os dados:
- Para o Spider: certifique-se de ter os arquivos `train.json`, `dev.json`, `tables.json` e o diretório `database/` em `data/spider/`.
- Para o MMLU: coloque `mmlu_150_eval.jsonl` em `data/mmlu_subset/`.

---

## 3. Pipeline de Execução End-to-End

### 🔹 Passo 1: Pré-processar os Dados do Spider
```bash
python scripts/preprocess_spider.py
```
> Gera `train_formatted.jsonl` e `dev_formatted.jsonl` em `data/spider/`.

---

### 🔹 Passo 2: Avaliar o Modelo Base (Baseline - sem Fine-Tuning)
```bash
python scripts/eval_spider.py --mode baseline --output_file results/baseline_spider_outputs.json --max_samples 20
```
> Mede a `ExecutionAccuracy` em 20 amostras. Relatório salvo em `results/baseline_spider_outputs_report.json`.

---

### 🔹 Passo 3: Treinar os Modelos com LoRA

#### Configuração 1:
```bash
python scripts/train_lora.py --config_file configs/lora_config_1.json
```

#### Configuração 2:
```bash
python scripts/train_lora.py --config_file configs/lora_config_2.json
```

> Adaptadores são salvos em:  
> `results/lora_config_1/final_adapter/`  
> `results/lora_config_2/final_adapter/`

---

### 🔹 Passo 4: Avaliar os Modelos Fine-Tuned (Text-to-SQL)

#### Avaliação da Configuração 1:
```bash
python scripts/eval_spider.py --mode finetuned --lora_adapter_path results/lora_config_1/final_adapter/ --output_file results/finetuned_config1_outputs.json --max_samples 20
```

#### Avaliação da Configuração 2:
```bash
python scripts/eval_spider.py --mode finetuned --lora_adapter_path results/lora_config_2/final_adapter/ --output_file results/finetuned_config2_outputs.json --max_samples 20
```

---

### 🔹 Passo 5: Medir a Regressão de Capacidade (MMLU)

#### Baseline no MMLU:
```bash
python scripts/eval_mmlu.py
```

#### Configuração 1 no MMLU:
```bash
python scripts/eval_mmlu.py --lora_adapter_path results/lora_config_1/final_adapter/
```

#### Configuração 2 no MMLU:
```bash
python scripts/eval_mmlu.py --lora_adapter_path results/lora_config_2/final_adapter/
```

---

## 📊 Objetivo Final

Com os resultados em mãos, você poderá comparar:

- **Ganho de especialização**: Aumento na `ExecutionAccuracy` no dataset Spider.
- **Perda de generalização**: Redução na performance no subset do MMLU.

Esses dados permitem uma análise crítica sobre o impacto da especialização em LLMs e o fenômeno do esquecimento catastrófico.

---

## 📌 Créditos

Este projeto foi desenvolvido por [Acauan C. Ribeiro](https://github.com/acauanrr) para fins de experimentação acadêmica no contexto de *fine-tuning* e avaliação de Large Language Models.

---