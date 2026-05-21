import subprocess
import sys
import os

# 测试能否直接导入 mineru.cli.client
print("=== Testing direct Python import ===")
try:
    result = subprocess.run(
        [sys.executable, '-c', 'from mineru.cli.client import main; print("Import OK")'],
        capture_output=True,
        text=True,
        timeout=30
    )
    print("stdout:", result.stdout)
    print("stderr:", result.stderr[:500] if result.stderr else "")
    print("returncode:", result.returncode)
except Exception as e:
    print(f"Error: {e}")

# 检查 mineru-lmdeploy-server 的帮助
print("\n=== mineru-lmdeploy-server help ===")
lmdeploy_path = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru-lmdeploy-server.exe'
if os.path.exists(lmdeploy_path):
    process = subprocess.Popen(
        [lmdeploy_path, '--help'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    start = time.time()
    while True:
        line = process.stdout.readline()
        if line:
            print(line.strip())
        if process.poll() is not None:
            break
        if time.time() - start > 10:
            process.kill()
            print("TIMEOUT")
            break