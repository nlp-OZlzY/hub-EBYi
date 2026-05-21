import mineru
import os

print("MinerU path:", mineru.__path__)

# 列出所有文件
for p in mineru.__path__:
    print(f"\nContents of {p}:")
    for item in os.listdir(p):
        full_path = os.path.join(p, item)
        if os.path.isfile(full_path):
            print(f"  [FILE] {item}")
        else:
            print(f"  [DIR] {item}/")