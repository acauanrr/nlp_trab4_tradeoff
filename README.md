# Projeto de Fine-Tuning de LLM para Text-to-SQL e Análise de Trade-off

Este projeto implementa um pipeline para fine-tuning do `Llama-3-8B-Instruct` na tarefa de Text-to-SQL usando o dataset Spider. Ele avalia o ganho de performance na tarefa específica e mede a regressão de capacidade em tarefas de conhecimento geral usando o MMLU.

## 1. Setup do Ambiente

**Pré-requisitos**:
- Python 3.9+
- NVIDIA GPU com suporte a CUDA e VRAM >= 16GB (para QLoRA)
- `git` e `git-lfs` instalados

**Passos**:

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd project_root
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Autentique-se no Hugging Face Hub:**
    (Você precisa de um token com acesso ao Llama-3)
    ```bash
    huggingface-cli login
    ```

5.  **Baixe os dados:**
    - Baixe o [dataset Spider](https://yale-lily.github.io/spider) e extraia os arquivos `train.json`, `dev.json`, `tables.json` e o diretório `database` para `data/spider/`.
    - Prepare seu subset de 150 questões do MMLU e coloque-o em `data/mmlu_subset/`.

## 2. Pipeline de Execução End-to-End

Execute os comandos na ordem especificada.

### Passo 1: Pré-processar os Dados do Spider

Este comando converte os dados do Spider para o formato de instrução do Llama-3.

```bash
python scripts/preprocess_spider.py \
    --train_path data/spider/train.json \
    --dev_path data/spider/dev.json \
    --tables_path data/spider/tables.json \
    --output_dir data/spider# nlp_trab4_tradeoff
