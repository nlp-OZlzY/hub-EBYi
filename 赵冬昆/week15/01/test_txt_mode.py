import subprocess
import time
import os

mineru_path = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru.exe'
pdf_path = r'D:\111111111111Study\BADOU\第15周：ClaudeCode、文档解析和DeepResearch\01homework1\uploads\09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf'
output_dir = r'D:\111111111111Study\BADOU\第15周：ClaudeCode、文档解析和DeepResearch\01homework1\processed'

os.makedirs(output_dir, exist_ok=True)

# 使用 txt 模式，不依赖 VLM
print("=== Testing mineru with txt method ===")
command = f'"{mineru_path}" -p "{pdf_path}" -o "{output_dir}" -m txt'

print(f"Command: {command}")

process = subprocess.Popen(
    command,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

start = time.time()
while True:
    line = process.stdout.readline()
    if line:
        print(line.strip())
    if process.poll() is not None:
        break
    if time.time() - start > 120:  # 2分钟超时
        print('TIMEOUT - killing process')
        process.kill()
        break

print('Process finished with return code:', process.returncode)