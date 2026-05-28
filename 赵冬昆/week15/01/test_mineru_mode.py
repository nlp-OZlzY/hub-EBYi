import subprocess
import time

mineru_path = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru.exe'
pdf_path = r'D:\111111111111Study\BADOU\第15周：ClaudeCode、文档解析和DeepResearch\01homework1\uploads\09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf'
output_dir = r'D:\111111111111Study\BADOU\第15周：ClaudeCode、文档解析和DeepResearch\01homework1\processed'

# 使用 vlm-http-client 模式，指定服务器URL
# 由于没有远程服务器，我们使用本地 vllm 服务器
# 但先检查是否有 vllm 可用

print("=== Testing mineru with vlm-http-client mode ===")
print("Note: This requires a running VLM server")
print("Command would be:")
print(f'"{mineru_path}" -p "{pdf_path}" -o "{output_dir}" -b vlm-http-client -u http://127.0.0.1:30000')

# 检查 mineru-lmdeploy-server
lmdeploy_path = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru-lmdeploy-server.exe'
print(f"\n=== Checking mineru-lmdeploy-server: {lmdeploy_path} ===")
import os
print(f"Exists: {os.path.exists(lmdeploy_path)}")

# 测试能否直接运行 Python 模块
print("\n=== Testing direct Python import ===")
result = subprocess.run(
    [sys.executable, '-c', 'from mineru.cli.client import main; print("Import OK")'],
    capture_output=True,
    text=True,
    timeout=30
)
print("stdout:", result.stdout)
print("stderr:", result.stderr[:500] if result.stderr else "")
print("returncode:", result.returncode)