"""RAG
第二步：文档切分（补充）—— MarkdownHeaderTextSplitter，按标题层级切。

【和长度切分的区别】
前面几种都是「按字数切」，这一种是「按结构切」：它只看 Markdown 的 # 标题，
从一个标题到下一个同级（或更高级）标题之间，算一片。

【白送的好处：标题会进 metadata】
这才是它最值钱的地方 —— 每片都带着自己所在的标题路径：

    原文：## 二、文档切分  →  ### 2.3 中文必须自定义 separators
    切片：{"二级标题": "二、文档切分", "三级标题": "2.3 中文必须自定义 separators"}

用户问「编码问题怎么解决」，召回的片段能直接说清出自哪一节，而不是只给一个文件名。

【两个注意点】
  1. 它【不管每片有多长】—— 只看标题。标题下面如果是一大段，切出来照样超长。
     真实项目通常再接一层 RecursiveCharacterTextSplitter 做二次切分。
  2. headers_to_split_on 要按层级写全。sample.md 里有三级标题，
     只写 # 和 ## 的话，### 会被当成普通正文，不进 metadata。

跑法：python 018.rag.splitter_MarkdownHeaderTextSplitter.py
"""
import json
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter

MARKDOWN_FILE = Path("assets/sample.md")

# 中文注释：元组的第二个元素是它写进 metadata 时用的键名，随便起，但起清楚点好。
HEADERS_TO_SPLIT_ON = [
    ("#", "一级标题"),
    ("##", "二级标题"),
    ("###", "三级标题"),
]

# 中文注释：正文完整打印会把 JSON 撑得没法看，这里截成预览。
PREVIEW_LIMIT = 120


def preview(text: str, limit: int = PREVIEW_LIMIT) -> str:
    """截断正文用于预览，超出时以省略号结尾（总长仍不超过 limit）。"""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def split_markdown(path: Path):
    """按标题层级切分 Markdown 文件，返回 list[Document]。"""

    text = path.read_text(encoding="utf-8")

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    chunks = splitter.split_text(text)
    print(f"原文 {len(text)} 字 -> {len(chunks)} 片\n")

    # 中文注释：Document 不是 JSON 可序列化的对象，得手动转成字典 —— 两样东西就是
    # page_content（正文）和 metadata（标题路径）。
    #
    # ensure_ascii=False 是必须的！它默认是 True，会把中文转成 "\u4e2d\u6587" 这种转义，
    # 看着像乱码（其实数据是对的，只是不可读，中文全变成编码）。
    payload = [
        {
            "page_content": preview(chunk.page_content),
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    return chunks


if __name__ == "__main__":
    split_markdown(MARKDOWN_FILE)
