import subprocess
import os

mineru_path = r'C:\Users\ilove\AppData\Local\Programs\Python\Python313\Scripts\mineru.exe'
pdf_path = os.path.abspath('./uploads/09b97e48-79ec-4d6e-95ce-9f3a92961ce7.pdf')
output_dir = os.path.abspath('./processed')

print('PDF exists:', os.path.exists(pdf_path))
print('PDF path:', pdf_path)
print('Output dir:', output_dir)

os.makedirs(output_dir, exist_ok=True)

command = f'{mineru_path} -p "{pdf_path}" -o "{output_dir}"'
print('Command:', command)

result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
print('Return code:', result.returncode)
print('stdout:', result.stdout)
print('stderr:', result.stderr)

# Check results
print('\nChecking output directory:')
if os.path.exists(output_dir):
    for item in os.listdir(output_dir):
        print(f'  {item}')