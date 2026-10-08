"""RAG
第一步：文档加载（Document Loading）—— 从 Excel（.xlsx）读取表格数据。
"""
import datetime
from pathlib import Path

import openpyxl
from langchain_core.documents import Document

# 中文注释：改成你自己的 xlsx 路径即可。这个文件只读，不会被修改。
XLSX_FILE = Path("assets/sample.xlsx")


def load_from_excel():
    """加载本地 .xlsx，返回 list[Document]。

    中文注释：注意粒度是【一数据行一个 Document】，不是整个表一个。
    txt / pdf / word 是「一篇文章整份读」，但表格是「多行独立记录」，
    检索时用户要的是「符合某个条件的那一行」，把整表塞成一个 Document 会把
    每条记录的语义稀释掉，命中了也不知道是哪一行。所以逐行拆开。
    """

    if not XLSX_FILE.exists():
        print(f"找不到 {XLSX_FILE.resolve()}")
        return []

    # 中文注释：data_only=True 读的是单元格当前值，而不是公式原文 —— 不加它会读到
    # '=SUM(B2:B10)' 这种字符串，对检索毫无意义。
    # 代价：它读的是「最后一次由 Excel / WPS 保存时缓存的公式结果」，
    # 如果文件是由程序生成的、从没被表格软件打开过，公式单元格可能读出 None。
    workbook = openpyxl.load_workbook(XLSX_FILE, data_only=True)

    documents = []
    for sheet in workbook.worksheets:
        # 中文注释：values_only=True 直接拿值，不用管 Cell 对象。
        # 行号按 Excel 的习惯从 1 开始，方便回表里核对。
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue

        headers = rows[0]
        for row_index, values in enumerate(rows[1:], start=2):
            pairs = build_pairs(headers, values)
            if not pairs:
                continue  # 整行都是空的，跳过（你的表里第 4 行就是这种）

            documents.append(
                Document(
                    page_content="\n".join(f"{label}：{value}" for label, value in pairs),
                    metadata={
                        "source": str(XLSX_FILE),
                        "sheet": sheet.title,
                        "row": row_index,
                    },
                )
            )

    print(f"共 {len(documents)} 个 Document（每个数据行一个）")
    for doc in documents[:3]:
        meta = doc.metadata
        print(f"\n  --- {meta['sheet']} 第 {meta['row']} 行 ---")
        print(f"  {doc.page_content}")

    return documents


def build_pairs(headers: tuple, values: tuple) -> list[tuple[str, str]]:
    """把一行数据配上表头，得到 [(字段名, 值)]，并跳过空单元格。

    中文注释：不能无脑 zip(headers, values) —— 你的表 A 列整列是空的，
    无脑拼会产出「None：None」这种垃圾文本，白白污染向量检索。
    所以按「值是否为空」过滤，而不是按列位置。
    """
    pairs = []
    for column_index, value in enumerate(values):
        if value is None:
            continue

        label = headers[column_index] if column_index < len(headers) else None
        # 中文注释：表头本身也可能是空的（比如这份表的 A 列），
        # 这时用列号兜底，保证内容不丢。
        label = str(label) if label is not None else f"第{column_index + 1}列"
        pairs.append((label, format_value(value)))

    return pairs


def format_value(value) -> str:
    """把单元格的值转成适合喂给模型的文本。

    中文注释：两类值需要特别处理，否则拼出来的文本会很别扭。
    """
    # 中文注释：openpyxl 读日期给的是 datetime 对象，直接 str() 会带出 "00:00:00"，
    # 而表里显示的只是日期。日期在检索里是重要的筛选维度，格式干净更好用。
    if isinstance(value, datetime.datetime):
        if value.time() == datetime.time(0, 0):
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, datetime.date):
        return value.strftime("%Y-%m-%d")

    # 中文注释：Excel 的数字一律是浮点，1 读出来是 1.0。整数就去掉小数点，
    # 让文本贴近你在表里看到的那个样子。
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


if __name__ == "__main__":
    load_from_excel()
