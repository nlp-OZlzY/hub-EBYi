"""Document Processor Worker."""
import os
import sys
import glob
import subprocess
import time
import yaml

from models.orm import list_files, update_file_state, FileState, Session
from models.orm import File as FileModel
from services.embedding import get_embedding_service
from services.vectorstore import get_vector_store

_config = None


def load_config():
    global _config
    if _config is None:
        with open("config.yaml", "r") as f:
            _config = yaml.safe_load(f)
    return _config


def split_text2chunks(lines, chunk_size=256):
    """文本拆分为chunk"""
    chunks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "# References":
            continue
        if len(line) > 2 and line[0] == "[" and line[1].isdigit():
            continue

        if not chunks:
            chunks.append(line)
        else:
            if len(chunks[-1]) <= chunk_size:
                chunks[-1] += "\n" + line
            else:
                chunks.append(line)
    return chunks


def encode_document(markdown_path: str, file_id: int, file_name: str, file_path: str):
    """编码文档"""
    emb_service = get_embedding_service()
    vs = get_vector_store()

    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = split_text2chunks(lines)
    for chunk in chunks:
        try:
            # 分别提取纯文本和图片引用
            text_with_no_image = "\n".join([l for l in chunk.split("\n") if not l.startswith("![")])
            text_with_image = [l for l in chunk.split("\n") if l.startswith("![")]

            # BGE向量
            text_bge_emb = emb_service.encode_text(text_with_no_image)

            # CLIP文本向量
            text_clip_emb = emb_service.encode_text_for_clip(text_with_no_image)

            # CLIP图像向量
            if text_with_image:
                img_ref = text_with_image[0].split("](")[1].split(")")[0]
                img_path = os.path.dirname(markdown_path) + "/" + img_ref.split("/")[-1]
                img_clip_emb = emb_service.encode_image(img_path)
            else:
                img_clip_emb = [0.0] * 1024

            data = [{
                "text_vector": text_bge_emb,
                "clip_text_vector": text_clip_emb,
                "clip_image_vector": img_clip_emb,
                "text": chunk,
                "db_id": file_id,
                "file_name": file_name,
                "file_path": file_path
            }]
            vs.insert(data)
        except Exception as e:
            print(f"Error encoding chunk: {e}")


def process_file(file: FileModel):
    """处理单个文件"""
    file_id = file.id
    file_name = file.filename
    file_path = file.filepath

    print(f"Processing: {file_name}")

    # 更新状态
    update_file_state(file_id, FileState.PROCESSING)

    try:
        cfg = load_config()
        mineru_cfg = cfg["mineru"]
        output_dir = cfg["paths"]["processed_dir"]

        # 解析PDF
        base_name = os.path.basename(file_path).split(".")[0]
        cmd = f"{mineru_cfg['cmd']} -p {file_path} -o {output_dir} -b {mineru_cfg['backend']} -u {mineru_cfg['url']}"

        subprocess.check_output(cmd, shell=True, timeout=600)

        # 查找markdown
        md_paths = glob.glob(os.path.join(output_dir, base_name, "**", "*.md"))
        if not md_paths:
            raise ValueError(f"No markdown found for {file_name}")

        # 编码
        encode_document(md_paths[0], file_id, file_name, file_path)

        # 完成
        update_file_state(file_id, FileState.COMPLETED)
        print(f"Completed: {file_name}")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        update_file_state(file_id, FileState.FAILED)


def main():
    """主循环"""
    cfg = load_config()
    interval = cfg["worker"]["interval_seconds"]

    print(f"Worker started, checking every {interval}s...")

    while True:
        try:
            with Session() as session:
                files = session.query(FileModel).filter(
                    FileModel.filestate == FileState.PENDING
                ).all()

            for f in files:
                process_file(f)

        except Exception as e:
            print(f"Worker error: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    main()