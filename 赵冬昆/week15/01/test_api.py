import subprocess
import time
import requests
import os

# 启动 mineru-api.exe 作为服务器
api_exe = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru-api.exe'

print("Starting mineru-api server...")
print(f"Executable: {api_exe}")
print(f"Exists: {os.path.exists(api_exe)}")

# 先测试 --help
print("\nTesting mineru-api --help:")
process = subprocess.Popen(
    [api_exe, '--help'],
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
    if time.time() - start > 30:
        print('TIMEOUT')
        process.kill()
        break

print("\nReturn code:", process.returncode)