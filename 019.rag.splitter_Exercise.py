"""RAG
综合练习：文档切分（补充）—— 按中文的「段落 / 句子 / 断句」三级切分。

【原理就一句话】
级数由 separators 的排列顺序决定。切分器从上往下试，上一级切完还超 chunk_size
就降到下一级 —— 所以把中文的层级符号按「颗粒从大到小」排好，它就自然往细里切：

    \\n\\n          ① 段落
    \\n            ② 换行
    。！？          ③ 句子（句末标点）
    ；，            ④ 断句（句中标点）

【怎么控制切到哪一级】
调 chunk_size 就行，不用改 separators：
    chunk_size 大  ->  停在①，每片是一个完整段落
    chunk_size 中等 ->  降到③，每片是一到两句
    chunk_size 小  ->  降到④，每片是一个分句
这正是「递归」的含义：它总是先试粗颗粒，装不下才往细里降。

跑法：python 019.rag.splitter_Exercise.py
"""
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_FILE = Path("assets/sample.txt")

# 中文注释：顺序就是颗粒从大到小，最后两项是中文特有的 —— 英文用空格分词，用不上。
SEPARATORS = [
    "\n\n",           # ① 段落
    "\n",             # ② 换行
    "。", "！", "？",  # ③ 句子：句末标点
    "；", "，",        # ④ 断句：句中标点
]

# 中文注释：同一段文字，只改 chunk_size，就能看到切口停在哪一级。
LEVELS = [
    ("① 段落级", 2000),
    ("② 句子级", 1200),
    ("③ 断句级", 450),
]

def split_chinese(text: str, chunk_size: int) -> list[str]:
    """按中文的段落 / 句子 / 断句三级切分，返回 list[str]。"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=SEPARATORS,
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    text = TEXT_FILE.read_text(encoding="utf-8").strip()
    print(f"原文 {len(text)} 字\n")

    for label, chunk_size in LEVELS:
        chunks = split_chinese(text, chunk_size)
        lengths = [len(chunk) for chunk in chunks]

        print(f"【{label}】chunk_size={chunk_size} -> {len(chunks)} 片，长度 {min(lengths)}~{max(lengths)}")
        # 中文注释：只看前两片就能看出切口落在哪 —— 段落级断在空行，句子级断在句号，断句级断在逗号。
        for index, chunk in enumerate(chunks[:2], start=1):
            print(f"    [{index}] {chunk.strip()[:600]}…")
        print()
