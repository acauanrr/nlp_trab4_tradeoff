import tensorflow as tf
import torch

print("===== VERIFICAÇÃO DE GPU =====\n")

# TensorFlow
print("[TensorFlow]")
try:
    print("Versão:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow detectou {len(gpus)} GPU(s):")
        for gpu in gpus:
            print("  →", gpu)
    else:
        print("❌ Nenhuma GPU detectada pelo TensorFlow.")
except Exception as e:
    print("Erro ao verificar GPU no TensorFlow:", e)

print("\n[PyTorch]")
# PyTorch
try:
    print("Versão:", torch.__version__)
    if torch.cuda.is_available():
        print(f"✅ PyTorch detectou uma GPU: {torch.cuda.get_device_name(0)}")
        print("Memória total da GPU:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
    else:
        print("❌ Nenhuma GPU detectada pelo PyTorch.")
except Exception as e:
    print("Erro ao verificar GPU no PyTorch:", e)
