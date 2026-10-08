"""RAG
第一步：文档加载（Document Loading）—— 从文本文件加载内容。
"""
from pathlib import Path

# 中文注释：副作用导入 —— 这个模块在被导入时设置 UA 环境变量、屏蔽 langchain-community
# 的 sunset 警告。langchain_community 是「导入时」就去读这些的，所以必须排在它前面。
# noqa: F401 是必要的：这个 import 确实没被使用，它的作用在执行导入本身。
import common.setup  # noqa: F401
from langchain_community.document_loaders import TextLoader

# 中文注释：改成你自己的 txt 路径即可。这个文件只读，不会被覆盖或删除。
TXT_FILE = Path("assets/sample.txt")


def load_from_txt():
    """加载本地 txt，返回 list[Document]。"""

    if not TXT_FILE.exists():
        print(f"找不到 {TXT_FILE.resolve()}")
        return []

    # 中文注释：encoding 一定要显式写 utf-8。不写就用系统默认编码 ——
    # 普通中文 Windows 默认是 cp936，读 UTF-8 的中文文件会乱码，甚至直接抛
    # RuntimeError: Error loading xxx.txt。
    docs = TextLoader(str(TXT_FILE), encoding="utf-8").load()

    print(f"共 {len(docs)} 个 Document")
    for doc in docs:
        print(f"  source = {doc.metadata['source']}")
        print(f"  正文   = {doc.page_content[:60]}…")

    return docs


if __name__ == "__main__":
    load_from_txt()
