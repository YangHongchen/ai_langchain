"""RAG
第一步：文档加载（Document Loading）—— 从 PDF 读取文字。
"""
from pathlib import Path

# 中文注释：副作用导入 —— 这个模块在被导入时设置 UA 环境变量、屏蔽 langchain-community
# 的 sunset 警告。langchain_community 是「导入时」就去读这些的，所以必须排在它前面。
# noqa: F401 是必要的：这个 import 确实没被使用，它的作用在执行导入本身。
import common.setup  # noqa: F401
from langchain_community.document_loaders import PyPDFLoader

# 中文注释：改成你自己的 pdf 路径即可。这个文件只读，不会被修改。
PDF_FILE = Path("assets/sample2.pdf")


def load_from_pdf():
    """加载本地 PDF，返回 list[Document]。"""

    if not PDF_FILE.exists():
        print(f"找不到 {PDF_FILE.resolve()}")
        return []

    # 中文注释：PDF 没必要传 encoding —— 它是二进制格式，文字是自己内部编码的，
    # 不像 txt 那样依赖外部字符编码。
    #
    # 和 TextLoader 最大的区别：PyPDFLoader 默认【每页一个 Document】。
    # txt 是一个文件一个 Document，而一份 100 页的 PDF 会得到 100 个 Document，
    # metadata 里的 page 就是页码（从 0 开始）。
    docs = PyPDFLoader(str(PDF_FILE)).load()

    total_chars = sum(len(doc.page_content.strip()) for doc in docs)
    print(f"共 {len(docs)} 个 Document，合计 {total_chars} 字")

    for doc in docs[:3]:
        meta = doc.metadata
        print(f"\n  第 {meta['page_label']} 页 / 共 {meta['total_pages']} 页")
        print(f"  source = {meta['source']}")
        print(f"  正文   = {doc.page_content[:80]}…")

    # 中文注释：最常见的坑 —— 扫描版 PDF（整页是一张图片）抽出来全是空字符串。
    # 判断方法：文件很大但总字数接近 0。这种必须先 OCR 才能拿到文字。
    if total_chars < 10:
        print("\n  注意：几乎没抽到文字 —— 大概率是扫描版 PDF，需要 OCR。")

    return docs


if __name__ == "__main__":
    load_from_pdf()
