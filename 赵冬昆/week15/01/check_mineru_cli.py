import os

mineru_cli = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Lib\site-packages\mineru\cli'
print(f"CLI directory: {mineru_cli}")

if os.path.exists(mineru_cli):
    for item in os.listdir(mineru_cli):
        full_path = os.path.join(mineru_cli, item)
        print(f"  {item}")
        if os.path.isdir(full_path):
            for subitem in os.listdir(full_path):
                print(f"    {subitem}")