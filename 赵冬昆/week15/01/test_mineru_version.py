import subprocess
import time
import sys

# 尝试使用 Python 直接调用 mineru 的 API
print("=== Testing MinerU Python API ===")

# 首先设置环境变量禁用某些可能阻塞的模块
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# 尝试导入并查看可用的函数
result = subprocess.run(
    [sys.executable, '-c', '''
import sys
sys.stdout.flush()
print("Starting import...", flush=True)

# 尝试快速导入
try:
    from mineru.version import __version__
    print(f"MinerU version: {__version__}", flush=True)
except Exception as e:
    print(f"Error importing version: {e}", flush=True)
    sys.exit(1)

print("Import successful!", flush=True)
'''],
    capture_output=True,
    text=True,
    timeout=30
)
print("stdout:", result.stdout)
print("stderr:", result.stderr[:1000] if result.stderr else "")
print("returncode:", result.returncode)