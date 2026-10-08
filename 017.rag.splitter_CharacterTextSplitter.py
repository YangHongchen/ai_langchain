"""RAG
第二步：文档切分（补充）—— CharacterTextSplitter，以及它为什么很少用了。

【一句话区别】
    CharacterTextSplitter   只按【一个】分隔符切，切不动就摆烂。
    RecursiveCharacter...   按【一张从粗到细的清单】逐级降级地切。
差别就在「递归」两个字：前者没有降级机制。

【它的两个坑】
  1. 参数是 separator（单数、字符串），不是 separators（列表）。写成列表会直接报错，和 Recursive 的签名不兼容。
  2. 找不到那个分隔符时【静默失效】：不报错、不警告，
     直接把整段当作一片返回，哪怕它远超 chunk_size。实测：
         CharacterTextSplitter(chunk_size=50).split_text('一' * 200)
         -> 1 片，200 字        ← chunk_size 被完全无视，而且没有任何提示
     这是最危险的地方 —— 你以为参数生效了，其实没有。

【为什么现在基本不用它】
真实文档里换行、空行的分布完全不可控。指望「每段之间都有 \\n\\n」太脆弱，
一旦没有，chunk_size 就形同虚设，而你还察觉不到。

只有文本【结构高度规整、某个分隔符一定存在】时它才合适，比如
「一行一条的日志」—— separator='\\n' 就够了，也不需要降级。

【顺带一个默认值差异】
    CharacterTextSplitter   keep_separator 默认 False —— 切完把分隔符丢掉
    RecursiveCharacter...   keep_separator 默认 True  —— 分隔符留在片头
所以同一段文本，Recursive 切出来是「，我国在太原…」，Character 切出来是「我国在太原…」。

跑法：python 017.rag.splitter_CharacterTextSplitter.py
"""
from pathlib import Path

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

TXT_FILE = Path("assets/sample.txt")


def show_chunks(title: str, chunks: list[str]) -> None:
    """打印切片结果，超过 chunk_size 的标出来。"""
    print(f"\n===== {title} —— 共 {len(chunks)} 片 =====")
    for index, chunk in enumerate(chunks, start=1):
        oversize = "   <<< 超过 chunk_size=30" if len(chunk) > 30 else ""
        print(f"  [{index}] {len(chunk):>3} 字 | {chunk}{oversize}")


def main() -> None:
    # 中文注释：用 Path 直接读，不经过 Document —— 这里关注的是切分本身，
    # split_text() 直接吃字符串、吐字符串列表，看切口最清楚。
    text = TXT_FILE.read_text(encoding="utf-8").strip()
    print(f"原文 {len(text)} 字：\n  {text}")

    # ---------- 坑一：默认 separator 是 '\n\n' ----------
    # 中文注释：这段文本一个换行都没有，所以 '\n\n' 根本找不到。
    # 结果不是报错，而是「整段当成一片」返回 —— 静默失效。
    default_splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=0)
    show_chunks(
        "坑一：默认 separator='\\n\\n'，而文本里没有换行",
        default_splitter.split_text(text),
    )

    # ---------- 坑二：改对了分隔符就正常，但只认这一个 ----------
    # 中文注释：指定中文逗号后切片正常了。代价是它【只会】按逗号切，
    # 遇到句号、分号、换行都不管 —— 这就是「没有降级机制」的含义。
    comma_splitter = CharacterTextSplitter(separator="，", chunk_size=30, chunk_overlap=0)
    show_chunks(
        "指定 separator='，' 之后",
        comma_splitter.split_text(text),
    )

    # ---------- 对照：Recursive 的表现 ----------
    # 中文注释：同一段文本、同样的 chunk_size，Recursive 先用 ' ' 切（TEXT 后面那个空格），
    # 不够小再降到逐字符硬切 —— 虽然切口未必理想，但它【保证】每片都不超 chunk_size。
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)
    show_chunks(
        "对照：RecursiveCharacterTextSplitter（默认 separators）",
        recursive_splitter.split_text(text),
    )

    print("\n===== 小结 =====")
    print("  Character 找不到分隔符 -> 1 片 91 字，chunk_size 被无视")
    print("  Recursive 找不到粗颗粒 -> 降级继续切，每片都 <= chunk_size")
    print("  所以现在默认选 RecursiveCharacterTextSplitter")


if __name__ == "__main__":
    main()
