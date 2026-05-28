import pdfplumber
import os

# --- 配置 ---
PDF_FILE = "汽车知识手册.pdf"
OUTPUT_MD = "汽车知识手册_parsed.md"


def parse_to_markdown(pdf_path, output_path):
    if not os.path.exists(pdf_path):
        print(f"❌ 错误：找不到文件 {pdf_path}")
        return

    print(f"🚀 正在使用 pdfplumber 解析并转换为 Markdown: {pdf_path} ...")

    with pdfplumber.open(pdf_path) as pdf, open(output_path, "w", encoding="utf-8") as f:
        total_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            print(f"正在处理第 {i + 1}/{total_pages} 页...")

            # 写入页眉
            f.write(f"\n\n### 第 {i + 1} 页\n\n")

            # --- 1. 提取表格 ---
            # extract_tables 会把表格解析为二维数组
            tables = page.extract_tables()

            # 记录表格占据的区域，防止文本重复提取
            table_rects = []

            for table in tables:
                # 简单过滤：如果表格第一行是空的，可能是误检，跳过
                if not table or not table[0]:
                    continue

                # 写入 Markdown 表格语法
                for r_idx, row in enumerate(table):
                    # 过滤掉全为空的行
                    if not any(cell for cell in row if cell and cell.strip()):
                        continue

                    # 清理单元格内容：去除换行符，防止破坏 Markdown 表格结构
                    cleaned_row = [
                        (cell.strip().replace('\n', ' ') if cell else "")
                        for cell in row
                    ]

                    # 写入行数据
                    f.write("| " + " | ".join(cleaned_row) + " |\n")

                    # 如果是第一行（表头），下面需要加分割线
                    if r_idx == 0:
                        f.write("|" + "|".join(["---"] * len(cleaned_row)) + "|\n")

                f.write("\n")  # 表格之间留空行

            # --- 2. 提取普通文本 ---
            # extract_text() 会尽量保持原始的排版顺序
            text = page.extract_text(
                x_tolerance=1,
                y_tolerance=1,
                layout=True  # 保持布局，这对双栏文档很重要
            )

            if text:
                f.write(text)
                f.write("\n\n")

    print(f"✅ 解析完成！Markdown 文件已保存为: {output_path}")


if __name__ == "__main__":
    parse_to_markdown(PDF_FILE, OUTPUT_MD)