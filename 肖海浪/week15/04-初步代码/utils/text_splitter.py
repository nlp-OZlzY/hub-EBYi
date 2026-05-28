"""文本切分工具"""
from typing import List


def split_text2chunks(lines: List[str], chunk_size: int = 256) -> List[str]:
    """
    将文本分割成多个块，每个块的长度不超过chunk_size个字符

    Args:
        lines: 文本行列表
        chunk_size: 每块最大字符数

    Returns:
        切分后的文本块列表
    """
    chunks = []

    for line in lines:
        line = line.strip()

        # 跳过空行
        if not line:
            continue

        # 跳过 References 标题
        if line == "# References":
            continue

        # 跳过引用行 [1] xxx
        if len(line) > 2:
            if line[0] == "[" and line[1].isdigit():
                continue

        # 合并或新建chunk
        if len(chunks) == 0:
            chunks.append(line)
        else:
            if len(chunks[-1]) <= chunk_size:
                chunks[-1] += "\n" + line
            else:
                chunks.append(line)

    return chunks
