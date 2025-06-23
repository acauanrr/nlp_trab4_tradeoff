# Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs via Fine-Tuning

Este repositório contém o código e os resultados do quarto trabalho prático para as disciplinas ICC220 e PPGINF528 da Universidade Federal do Amazonas (UFAM).

**Aluno**: Acauan C. Ribeiro

---

## 1. Objetivo do Projeto

O objetivo central deste projeto foi avaliar empiricamente o processo de fine-tuning em Modelos de Linguagem de Grande Porte (LLMs) para a tarefa de Text-to-SQL. A análise quantifica o ganho de desempenho na tarefa-alvo (usando o dataset Spider) e, simultaneamente, mede a alteração de performance em tarefas de conhecimento geral (usando o dataset MMLU), investigando o trade-off de especialização vs. generalização.

---

## 2. Estrutura do Repositório

```
.
├── configs/
│   ├── lora_config_1.json           # Hiperparâmetros para o 1º treino
│   └── lora_config_2.json           # Hiperparâmetros para o 2º treino
├── custom_metrics/
│   └── execution_accuracy.py  # Métrica customizada para DeepEval
├── notebooks/
│   └── nlp_proj4_new.ipynb          # Notebook Colab utilizado para orquestrar os experimentos
├── data/
│   ├── spider/
│   │   ├── train_formatted.jsonl  # Dataset de treino pré-processado
│   │   └── ... (outros arquivos do Spider)
│   └── mmlu_subset/
│       └── mmlu_150_eval.jsonl    # Subset de avaliação do MMLU
├── results/
│   ├── lora_config_1/             # Artefatos do treino 1 (checkpoints, etc)
│   ├── lora_config_2/             # Artefatos do treino 2 (checkpoints, etc)
│   └── ... (arquivos .json com as predições)
├── scripts/
│   ├── preprocess_spider.py     # Script para formatar o dataset Spider
│   ├── train_lora.py            # Script para o fine-tuning com LoRA
│   ├── eval_spider.py           # Script para avaliação no Spider com a métrica customizada
│   └── eval_mmlu.py             # Script para avaliação no MMLU
├── data.zip                     # Arquivo que deve ser descompactado /data
├── requirements.txt             # Dependências do projeto
└── README.md                    # Este arquivo
```

---

## 3. Como Reproduzir os Resultados

Para reproduzir todos os experimentos, siga os passos abaixo. Recomenda-se o uso de um ambiente com GPU (e.g., Google Colab com GPU T4 ou superior).

### Passo 1: Configuração do Ambiente

**Clone o repositório:**

```bash
git clone https://github.com/acauanrr/nlp_trab4_tradeoff.git
cd nlp_trab4_tradeoff
```

**Instale as dependências com as versões exatas para garantir a reprodutibilidade:**

```bash
pip install -r requirements.txt
```

**Faça o download e descompacte os dados do Spider:**

```bash
# O arquivo data.zip já está no repositório
unzip -q data.zip -d .
```

**Autentique-se no Hugging Face para baixar o modelo Llama-3:**

*No seu script ou notebook, execute:*

```python
from huggingface_hub import notebook_login
notebook_login()
# Cole seu token de acesso quando solicitado
```

---

### Passo 2: Pré-processamento dos Dados

Formate o dataset Spider para o padrão de chat utilizado no treinamento.

```bash
python scripts/preprocess_spider.py
```

Isso irá gerar o arquivo `data/spider/train_formatted.jsonl`, que será usado na próxima etapa.

---

### Passo 3: Treinamento (Fine-Tuning)

Foram testadas duas configurações de hiperparâmetros distintas. Para treinar cada modelo, execute os seguintes comandos:

**Treinamento com a Configuração 1 (`lora_config_1.json`):**

```bash
python scripts/train_lora.py     --config_file lora_config_1.json     --output_base_dir results     --maxsteps 2048
```

**Treinamento com a Configuração 2 (`lora_config_2.json`):**

```bash
python scripts/train_lora.py     --config_file lora_config_2.json     --output_base_dir results     --maxsteps 2048
```

Ao final, os adaptadores LoRA estarão salvos em `results/lora_config_1/final_adapter` e `results/lora_config_2/final_adapter`.

---

### Passo 4: Avaliação

#### 4.1 Avaliação na Tarefa-Alvo (Spider)

Para avaliar o modelo base e os modelos fine-tuned no dataset Spider, execute os seguintes comandos. Os resultados da métrica **Execution Accuracy** serão impressos no console, e as predições em SQL serão salvas nos arquivos `.json` especificados.

**Avaliação do Modelo Base:**

```bash
python scripts/eval_spider.py     --mode baseline     --output_file results/spider_baseline_preds.json     --batch_size 8
```

**Avaliação do Modelo Fine-Tuned (Config 1):**

```bash
python scripts/eval_spider.py     --mode finetuned     --lora_adapter_path results/lora_config_1/final_adapter     --output_file results/spider_finetuned_preds_1.json     --batch_size 8
```

**Avaliação do Modelo Fine-Tuned (Config 2):**

```bash
python scripts/eval_spider.py     --mode finetuned     --lora_adapter_path results/lora_config_2/final_adapter     --output_file results/spider_finetuned_preds_2.json     --batch_size 8
```

#### 4.2 Avaliação de Generalização (MMLU)

Para medir a regressão (ou ganho) de capacidade, avalie os três modelos no nosso subset do MMLU.

**Avaliação do Modelo Base:**

```bash
python scripts/eval_mmlu.py
```

**Avaliação do Modelo Fine-Tuned (Config 1):**

```bash
python scripts/eval_mmlu.py --lora_adapter_path results/lora_config_1/final_adapter
```

**Avaliação do Modelo Fine-Tuned (Config 2):**

```bash
python scripts/eval_mmlu.py --lora_adapter_path results/lora_config_2/final_adapter
```

---

## 4. Análise dos Resultados

A análise completa dos resultados, incluindo as tabelas comparativas e a discussão sobre o trade-off, está detalhada no relatório técnico em PDF. Os principais achados são:

- **Ganho de Especialização**:  
  O fine-tuning com LoRA resultou em um ganho massivo de performance na tarefa de Text-to-SQL, com a acurácia de execução saltando de **9.17% (modelo base)** para **60.83% (ambos os modelos fine-tuned)**.

- **Trade-off de Generalização**:  
  Surpreendentemente, não foi observado o fenômeno de *"esquecimento catastrófico"*. Pelo contrário, ambos os modelos fine-tuned demonstraram uma melhora na performance no teste de conhecimento geral MMLU.  
  O modelo da `config_1` (r=16) teve um ganho de acurácia de **+261%**, enquanto o da `config_2` (r=32) teve um ganho de **+185%** em relação ao baseline.

Esses resultados sugerem que, para o **Llama-3** em conjunto com a técnica **PEFT LoRA**, a especialização em uma tarefa de raciocínio complexo como **Text-to-SQL** pode, na verdade, **aprimorar as capacidades lógicas gerais do modelo**, em vez de degradá-las.
