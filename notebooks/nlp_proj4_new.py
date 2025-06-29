# -*- coding: utf-8 -*-
"""nlp-proj4-new.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EOtC8O84xTV0062KKlmNGfNHs5b_WbFV

## **Trabalho 04 - Análise Quantitativa do Trade-off**
### Aluno: **Acauan C. Ribeiro**

### **1. Setup Projeto**
"""

# Célula 1.1: Instalar as dependências do projeto
# Usamos -q para uma instalação mais limpa (quiet)
!pip install -q transformers datasets peft trl bitsandbytes accelerate deepeval einops

# Célula 1.3: Clonar o seu repositório do GitHub
!git clone https://github.com/acauanrr/nlp_trab4_tradeoff.git

import os

project_dir = "/content/nlp_trab4_tradeoff"
os.chdir(project_dir)

!pwd

# Confere tamanho
!ls -lh data.zip

# Descompacta
!unzip -q data.zip -d .

# Célula 1.1: Instalar as dependências do projeto
!pip install -q transformers datasets peft trl bitsandbytes accelerate deepeval einops

# Célula 1.2: Instalar o Git LFS no ambiente do Colab
!sudo apt-get install git-lfs

# Célula 1.3: Login no Hugging Face (necessário novamente após reiniciar)
from huggingface_hub import notebook_login

notebook_login()

# Célula de Treinamento (Executar DEPOIS de criar o arquivo)
import os

project_dir = "/content/nlp_trab4_tradeoff"
os.chdir(project_dir)

!pwd

"""### **2.1 Treino Spider - Modelo Lora 1**"""

# Commented out IPython magic to ensure Python compatibility.
# # Célula para CRIAR o arquivo de configuração
# %%writefile /content/nlp_trab4_tradeoff/lora_config_1.json
# {
#     "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
#     "lora_r": 16,
#     "lora_alpha": 32,
#     "lora_dropout": 0.05,
#     "learning_rate": 2e-5,
#     "num_epochs": 3,
#     "per_device_train_batch_size": 1,
#     "gradient_accumulation_steps": 4
# }

# Célula de Treinamento (Executar DEPOIS de criar o arquivo)
import os

project_dir = "/content/nlp_trab4_tradeoff"
os.chdir(project_dir)

!pwd

!python scripts/train_lora.py \
    --config_file lora_config_1.json \
    --dataset_path data/spider/train_formatted.jsonl \
    --output_base_dir results \
    --maxsteps 2048 \
    --seed 42

# Célula para compactar os resultados
# O comando -r significa "recursivo", para incluir todos os arquivos e subpastas
!zip -r lora1_results.zip results/lora_config_1

"""### **2.2 Treino Spider - Modelo Lora 2**"""

# Commented out IPython magic to ensure Python compatibility.
# # Célula para CRIAR o arquivo de configuração
# %%writefile /content/nlp_trab4_tradeoff/lora_config_2.json
# {
#     "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
#     "lora_r": 32,
#     "lora_alpha": 64,
#     "lora_dropout": 0.1,
#     "learning_rate": 1e-5,
#     "num_epochs": 3,
#     "per_device_train_batch_size": 1,
#     "gradient_accumulation_steps": 4
# }

# Célula de Treinamento (Executar DEPOIS de criar o arquivo)
import os

project_dir = "/content/nlp_trab4_tradeoff"
os.chdir(project_dir)

!pwd

!python scripts/train_lora.py \
    --config_file lora_config_2.json \
    --dataset_path data/spider/train_formatted.jsonl \
    --output_base_dir results \
    --maxsteps 2048 \
    --seed 42

# Célula para compactar os resultados
# O comando -r significa "recursivo", para incluir todos os arquivos e subpastas
!zip -r lora2_results.zip results/lora_config_2

import os

project_dir = "/content/nlp_trab4_tradeoff"
os.chdir(project_dir)

!pwd

"""### **3. Avaliar Baseline**"""

!python scripts/eval_spider.py \
  --mode baseline \
  --batch_size 4 \
  --max_samples 120 \
  --output_file results/spider_512.json

!deepeval view

"""### **3.1 Avaliar FineTunning - Lora 1**"""

!python scripts/eval_spider.py \
    --mode finetuned \
    --lora_adapter_path results/lora_config_1/final_adapter \
    --max_samples 120 \
    --batch_size 4 \
    --output_file results/spider_120_finetuned_1.json

"""### **3.2 Avaliar FineTunning - Lora 2**"""

!python scripts/eval_spider.py \
    --mode finetuned \
    --lora_adapter_path results/lora_config_2/final_adapter \
    --max_samples 120 \
    --batch_size 4 \
    --output_file results/spider_120_finetuned_2.json

"""### **4. Perda de Generalização no MMLU - Bseline**"""

# Célula de Autenticação
from huggingface_hub import notebook_login

notebook_login()

# Célula de Execução da Avaliação
!python scripts/eval_mmlu.py

"""### **4.1 Perda de Generalização no MMLU - Lora 1**"""

!python scripts/eval_mmlu.py --lora_adapter_path results/lora_config_1/final_adapter

"""### **4.2 Perda de Generalização no MMLU - Lora 2**"""

!python scripts/eval_mmlu.py --lora_adapter_path results/lora_config_2/final_adapter