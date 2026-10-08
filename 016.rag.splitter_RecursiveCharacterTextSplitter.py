"""RAG
第二步：文档切分（Text Splitting）—— 把长文档切成检索用的小块。

【为什么必须切】
  1. 塞不进 —— 上下文窗口有上限，长文档放不下
  2. 不精准 —— 用户问一个细节，召回的是整篇，向量被平均化后谁都不像
  3. 太费钱 —— 输入按 token 计费，每次都塞整篇等于重复付费

【三个参数：用「抄书换纸」来理解】
把一篇长文抄到多张纸上，抄满一张就换下一张：

    chunk_size    —— 一张纸能抄多少字。
    separators    —— 在哪下刀。聪明的话停在句号逗号处，别把词劈成两半。
    chunk_overlap —— 换纸时往回多抄几个字。
                     万一刀口正好落在一句话中间，前半句在上一张纸末尾、
                     后半句在下一张纸开头，两张纸各自都是残的。
                     往回多抄一段，至少有一张纸是完整的。

【separators：一张从粗到细的候选刀口清单】
程序从上往下试，上一级切完还超过 chunk_size 就降到下一级：
    \\n\\n 段落 → \\n 换行 → 。！？ 句末标点 → ；， 句中标点 → ' ' 空格 → '' 逐字符
最后的 '' 是兜底，必须留着 —— 否则超长又没标点的文本永远切不完。

LangChain 的默认清单是 ['\\n\\n', '\\n', ' ', '']，这是给【英文】设计的 ——
英文靠空格分词，' ' 是合理的下刀点。中文不用空格，默认清单里只有前两级能命中，
于是只能一路降级到逐字符硬切，把句子劈开。实测 sample.txt（91 字、无换行）：

    默认 separators：  ……在太原卫星发射中心使用长征六号 | 改运载火箭……
                       ← 把「长征六号改运载火箭」劈成两半
    中文 separators：  ……在太原卫星发射中心使用长征六号改运载火箭， | 成功……
                       ← 停在逗号，每一片都是完整的语义单元

所以中文必须自己写 separators：
    ["\\n\\n", "\\n", "。", "！", "？", "；", "，", " ", ""]

【调参顺序不能颠倒】
先 separators（让每刀落在标点上）→ 再 chunk_size（控制单片体量）
→ 最后 chunk_overlap（兜住切口）。
反过来就会一直在治标：切口本身就不对，加大 overlap 只是多存一份残句。

【取值参考】
    chunk_size      中文 300~800 字，一片 ≈ 一到两个自然段。
                    不是越小越准 —— 切太小会把一句话拆散，检索时反而匹配不上。
    chunk_overlap   取 chunk_size 的 10~20%，别超过 200。
                    设成和 chunk_size 一样大是最糟的：每片几乎全是上一片的内容，
                    等于同一段话存了一遍又一遍，检索质量反而下降。
    separators      中文必须自定义标点，见上。

注意：chunk_size 数的是【字符数】，不是 token 数。

【本例为什么用 chunk_size=30】
sample.txt 只有 91 字，用 500 去切只会得到 1 片，看不出效果。
故意设小，让切法肉眼可见。

跑法：python 016.rag.splitter_RecursiveCharacterTextSplitter.py
"""


from pathlib import Path

import common.setup  # noqa: F401
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

TXT_FILE = Path("assets/sample.txt")


def load_sample():
    """加载示例文档。

    中文注释：加载只是铺垫 —— 真正的主角是下面的切分，所以这里一行带过。
    """
    return TextLoader(str(TXT_FILE), encoding="utf-8").load()


def show_chunks(title: str, chunks: list) -> None:
    """打印切分结果，顺便标出每片的字数和首尾字符。"""
    print(f"\n===== {title} —— 共 {len(chunks)} 片 =====")
    for index, chunk in enumerate(chunks, start=1):
        content = chunk.page_content
        print(f"  [{index}] {len(content):>2} 字 | {content}")


def main() -> None:
    documents = load_sample()
    print(f"原文 {len(documents[0].page_content)} 字：")
    print(f"  {documents[0].page_content}")

    # ---------- 实验一：用默认 separators ----------
    # 中文注释：RecursiveCharacterTextSplitter 的默认分隔符是 ['\n\n', '\n', ' ', '']，
    # 这是【为英文设计的】—— 英文靠空格分词，所以 ' ' 是个合理的切分点。
    # 但中文不用空格分词，这段文本里唯一的空格在 “TEXT ” 后面，
    # 于是它只能一路降级到最后的 ''（按单个字符硬切），结果就是句子被劈开。
    default_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)
    show_chunks("实验一：默认 separators（chunk_size=30, chunk_overlap=0, separators=[英文默认]）", default_splitter.split_documents(documents))

    # ---------- 实验二：给中文加标点分隔符 ----------
    # 中文注释：中文的正确切分点是标点。把 separators 按「颗粒从大到小」排好，
    # 切分器会从上往下试：先找空行，再找换行，再找句号……实在不行才硬切。
    # 这样出来的每一片都停在标点上，是完整的语义单元。
    chinese_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,
        chunk_overlap=0,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
    )
    show_chunks("实验二：中文 separators（chunk_size=30, chunk_overlap=0, separators=[中文推荐]）", chinese_splitter.split_documents(documents))

    # ---------- 实验三：chunk_overlap 的作用 ----------
    # 中文注释：overlap 让相邻两片【重叠】若干字符。
    # 用途：切分点是人为的，万一正好切在一个关键的上下文中间，
    # 两片各自都不完整；重叠一段后，至少有一片能保留完整语义。
    # 代价：总字数和存储量变大（这里 5 片变 6 片），检索时也可能召回重复内容。
    overlap_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=10)
    show_chunks("实验三： chunk_overlap=10（chunk_size=30, chunk_overlap=10, separators=[英文默认]）", overlap_splitter.split_documents(documents))

    # ---------- 两个容易忽略的细节 ----------
    chunks = chinese_splitter.split_documents(documents)
    print(f"\n===== 两个细节 =====")
    # 中文注释：1. metadata 会自动继承 —— 每个 chunk 都知道自己来自哪个文件。
    # 这正是前面加载阶段辛苦保留 metadata 的意义：切片不会丢掉来源信息。
    print(f"  1. metadata 自动继承: {chunks[0].metadata}")

    # 中文注释：2. chunk_size 数的是【字符数】，不是 token 数。
    # 所以设 500 不代表「500 token」，中文的实际 token 数可能更多也可能更少 ——
    # 要精确控制就得把 length_function 换成 token 计数器。
    print(f"  2. chunk_size=30 实际切片最大长度: {max(len(c.page_content) for c in chunks)} 字")

if __name__ == "__main__":
    main()
