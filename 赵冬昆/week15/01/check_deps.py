import subprocess
import sys

# 检查 PyTorch 和 CUDA
print("=== Checking PyTorch ===")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed!")
except Exception as e:
    print(f"Error: {e}")

# 检查 transformers
print("\n=== Checking Transformers ===")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers not installed!")
except Exception as e:
    print(f"Error: {e}")

# 检查 vllm
print("\n=== Checking vLLM ===")
try:
    import vllm
    print(f"vLLM version: {vllm.__version__}")
except ImportError:
    print("vLLM not installed!")
except Exception as e:
    print(f"Error: {e}")

# 检查 GPU
print("\n=== Checking GPU with nvidia-smi ===")
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else result.stderr)