"""
测试配置文件 - pytest fixtures
"""

import pytest
import os
import tempfile
import shutil

# 设置测试环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

@pytest.fixture
def temp_db():
    """创建临时数据库"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

@pytest.fixture
def sample_pdf_path(temp_dir):
    """创建示例PDF文件"""
    pdf_path = os.path.join(temp_dir, "sample.pdf")
    # 创建一个空的PDF文件（实际测试时需要真实的PDF）
    with open(pdf_path, 'wb') as f:
        f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    return pdf_path

@pytest.fixture
def sample_markdown(temp_dir):
    """创建示例markdown文件"""
    md_path = os.path.join(temp_dir, "sample.md")
    content = """# 示例文档

这是第一段文本。

![image1](images/test1.jpg)

这是第二段文本，包含图片。

## 参考
[1] 参考资料1
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # 创建图片目录
    img_dir = os.path.join(temp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    return md_path, img_dir
