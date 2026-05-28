import subprocess
import time

mineru_path = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru.exe'

# 先测试 --help
print("Testing: mineru --help")
process = subprocess.Popen(
    [mineru_path, '--help'],
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