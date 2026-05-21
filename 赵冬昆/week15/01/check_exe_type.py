import subprocess
import os

# 使用 dumpbin 或 similar 来检查可执行文件类型
mineru_exe = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru.exe'

# 检查文件头
with open(mineru_exe, 'rb') as f:
    header = f.read(2)
    print(f"MZ header: {header}")  # MZ = Windows executable

# 使用 PowerShell 来获取文件信息
result = subprocess.run(
    ['powershell', '-Command', f'(Get-Item "{mineru_exe}").VersionInfo | Format-List'],
    capture_output=True,
    text=True
)
print("\nVersion Info:")
print(result.stdout)
print(result.stderr)