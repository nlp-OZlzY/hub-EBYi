import subprocess
import time
import os

# 检查 MinerU 可执行文件的依赖
mineru_exe = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru.exe'

print("File exists:", os.path.exists(mineru_exe))
print("File size:", os.path.getsize(mineru_exe) if os.path.exists(mineru_exe) else "N/A")

# 检查是否有 API server
print("\nChecking available MinerU tools:")
scripts_dir = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts'
for f in os.listdir(scripts_dir):
    if 'mineru' in f.lower():
        print(f"  {f}")

# 检查 mineru-api.exe 是否存在
api_exe = os.path.join(scripts_dir, 'mineru-api.exe')
if os.path.exists(api_exe):
    print(f"\nmineru-api.exe found at: {api_exe}")
    print("This might need to run as a server first!")
else:
    print("\nminer-u-api.exe not found")