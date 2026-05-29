# check_packages.py
import importlib
import sys

import dotenv
import kafka
import jose

# 从 requirements.txt 文件中读取包名
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 提取包名（去除版本号和注释）
    packages = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # 去除版本号约束，只保留包名
            package = line.split('>=')[0].split('==')[0].split('<')[0].split('[')[0]
            packages.append(package)

    return packages


# 检查包是否已安装
def check_packages(packages):
    missing_packages = []
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✅ 已安装: {package}")
        except ImportError as e:
            print(f"❌ 未安装: {package} (错误: {e})")
            missing_packages.append(package)

    return missing_packages


if __name__ == "__main__":
    print("🔍 正在检查依赖包...\n")

    # 读取 requirements.txt
    try:
        packages = read_requirements()
    except FileNotFoundError:
        print("❌ 错误: 找不到 requirements.txt 文件，请确保它在当前目录下。")
        sys.exit(1)

    # 检查包
    missing = check_packages(packages)

    if missing:
        print(f"\n⚠️ 发现 {len(missing)} 个包未安装。")
        print("建议运行以下命令安装：")
        print(f"pip install {' '.join(missing)}")
    else:
        print(f"\n🎉 恭喜！所有依赖包都已成功安装。")