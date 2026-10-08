"""RAG
文档切分（补充）—— SemanticChunker，按「语义」切。

【一句话原理】
先把文章拆成一个个「最小单元」，每个单元算一个向量；相邻单元意思差得远，就在中间切一刀。
所以它切的不是字数，是【话题】。

【关键在「最小单元」怎么定 —— 这是比调阈值更根本的一招】
默认它按句末标点切句。但在这份客服通话记录里，那样会出问题：客户的提问和客服的回答
是【两个单元】，语义边界一旦落在它们中间，一问一答就被劈进两片了 —— 这时候你怎么调阈值
都没用，因为问题出在单元的粒度上。

所以这里把单元改成【一问 + 一答】这个完整回合：在「客户：」前面断开（零宽断言，不消费
字符），每个单元正好是「客户问 … 客服答 …」。单元本身不可再分，语义切分就再也拆不开它们。

    🙋客户：我家宽带从昨天开始就特别慢…
    🎧客服：好的，我先帮您看一下线路状态…        ← 这两句是一个整体，永远在一起

换个场景同理：会议记录按「一人一次发言」，访谈按「问 + 答」，总之让单元落在业务上有意义
的完整边界上，而不是机械地落在句号上。

【中文还必须自己写正则】
它默认的正则只认英文的 . ? !，中文的 。！？ 一个都不认。后果很隐蔽：整篇中文会被当成
【一个单元】，函数直接原样返回 —— 跑得通、不报错、看着正常，实际上等于没切。

【什么时候用】
  用得动：长的、通篇一大段、话题来回跳的文字 —— 会议记录、通话录音转写、没有标题的制度文件。
  用不动：短文（几十句以内）样本太少，判断不稳；有标题或段落结构的文档，按结构切更准也更快。
  另外每个单元都要算一次向量，比按字数切慢得多，别拿它批量处理海量文档。

【阈值不用手调，脚本自己挑】
percentile 越大 = 门槛越高 = 切得越少、块越大。脚本从高往低试一串候选值，挑出第一个
「最长块不超过 MAX_CHUNK_LENGTH 字」的 —— 也就是在「不产生超长块」的前提下尽量少切。
但只管上限还不够：切得碎了会冒出「客户：好的。」这种几个字的碎片，进了向量库只会污染
召回。所以再给 min_chunk_size 兜一道下限，把过小的块并进下一块。

【两个小坑】
  1. 分块结果是用空格把单元拼回去的，中文句子之间会多出空格，展示时得先折平。
  2. 切出来的边界完全取决于 embedding 模型，换个模型结果就会变。

【这次演示】
sample_call.txt 是一份三千字的宽带客服通话记录，通篇没有小标题，正好是它擅长的类型。
输出按话轮分行，一眼就能看出一问一答有没有被拆散。

跑法：python 020.rag.splitter.SemanticChunker.py
"""
import re
from pathlib import Path

import common.setup  # noqa: F401
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.utils import build_embeddings

TEXT_FILE = Path("assets/sample_call.txt")

# 中文注释：语义切分的「最小单元」—— 在「客户：」前断开（零宽断言，不消费字符），
# 每个单元正好是【一问 + 一答】。这样一问一答天然不可分割，语义边界不可能落在它们中间。
# 如果换成默认的句末标点，问和答会变成两个单元，就会被拆散。
SEMANTIC_UNIT_REGEX = r"(?=客户：)"

# 中文注释：只用来「展示」的正则 —— 打印时在每句话前面断开，好让话轮一句一行。
TURN_SPLIT_REGEX = re.compile(r"(?=客服：|客户：)")

LENGTH_CHUNK_SIZE = 200

# 中文注释：中文标点分离器，颗粒从大到小。
# 最后那个 "" 是兜底，千万别删 —— 它是唯一能让切分器「降无可降就用单字切」的终止条件。
# 少了它，遇到「既没标点也没空格」的长串（订单号、URL、token）会直接产出超过 chunk_size
# 的块，而且不报错：源码里 new_separators 为空时会走 final_chunks.append(s) 原样收下。
LENGTH_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 中文注释：通话记录里「客服：」和「客户：」只差一个字，挤在一行根本分不清谁在说话，
# 给两边各配个图标，扫一眼就能认出来。项目里其他地方不写 emoji，这里是刻意用来做视觉区分的。
SPEAKER_ICONS = {
    "客服：": "🎧",
    "客户：": "🙋",
}

# 中文注释：单块的字数上限，同时是自动挑阈值的判据 —— 一块塞太多字，检索时召回就不准了。
MAX_CHUNK_LENGTH = 400

# 中文注释：单块的字数下限，交给 SemanticChunker 自己处理（注意它不是「丢掉」小块，
# 而是把小块并进后面那块）。没有它的话，切得碎的时候会掉出「客户：好的。」这种
# 几个字的碎片，进向量库只会污染召回。
MIN_CHUNK_LENGTH = 80

# 中文注释：候选阈值从高往低排（percentile 越大 = 门槛越高 = 切得越少），从 99 一路降到 34。
# 范围要够宽：单元越粗（比如这里的一问一答），能达到同一个「块长上限」的阈值就越低。
# 步长取 5 而不是逐个试，是因为每试一个值都要把所有单元重新算一遍向量。
AMOUNT_CANDIDATES = list(range(99, 29, -5))

def split_by_length(text: str, chunk_size: int = LENGTH_CHUNK_SIZE) -> list[str]:
    """对照组：按字数硬切（最常用的 RecursiveCharacterTextSplitter）。"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=LENGTH_SEPARATORS,
    )
    return splitter.split_text(text)


def split_by_semantic(text: str, embeddings: Embeddings, amount: float) -> list[str]:
    """按语义边界切分，返回 list[str]。

    中文注释：embeddings 从外面传进来，不在函数里构造 —— 加载 ONNX 模型有固定开销，
    每次调用都重新构造会白白多花好几秒。
    """

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=amount,
        sentence_split_regex=SEMANTIC_UNIT_REGEX,
        min_chunk_size=MIN_CHUNK_LENGTH,
    )
    return splitter.split_text(text)


def sweep_amounts(text: str, embeddings: Embeddings) -> list[tuple[float, list[str], int]]:
    """把候选阈值挨个试一遍。

    返回 [(阈值, 切片, 最长块字数), ...]，顺序和 AMOUNT_CANDIDATES 一致，即从高到低。
    """

    results = []
    for amount in AMOUNT_CANDIDATES:
        chunks = split_by_semantic(text, embeddings, amount)
        results.append((amount, chunks, max(len(chunk) for chunk in chunks)))
    return results


def pick_amount(results: list[tuple[float, list[str], int]]) -> tuple[float, list[str]]:
    """从扫描结果里挑出要用的那一次切分。

    中文注释：results 是从高往低排的，所以第一个满足「最长块不超上限」的，就是
    「切得最少、且没有超长块」的那次。如果全都不满足（文档里确实有一大段同话题的
    长内容），就退回最后一个 —— 也就是切得最碎的那次。
    """

    for amount, chunks, longest in results:
        if longest <= MAX_CHUNK_LENGTH:
            return amount, chunks

    amount, chunks, _ = results[-1]
    return amount, chunks


def split_turns(chunk: str) -> list[str]:
    """把一片拆成「话轮」列表，用于分行打印。

    中文注释：SemanticChunker 是用空格把单元拼回一片的，所以先折平空白，
    再在「客服：」「客户：」前面断开 —— 这样一句一行，一问一答看得清。
    """

    flat = " ".join(chunk.split())
    return [turn for turn in TURN_SPLIT_REGEX.split(flat) if turn]


def format_turn(turn: str) -> str:
    """给话轮加上发言人图标。

    中文注释：切口落在话轮中间时，开头那段没有「客服：」「客户：」前缀，用两个空格占位，
    免得整片看起来参差不齐。
    """

    for prefix, icon in SPEAKER_ICONS.items():
        if turn.startswith(prefix):
            return f"{icon}{turn}"
    return f"  {turn}"


def show(title: str, chunks: list[str]) -> None:
    """逐片打印，每片按话轮分行 —— 一眼看出一问一答有没有被拆散。"""

    lengths = [len(chunk) for chunk in chunks]
    print(f"\n【{title}】{len(chunks)} 片，长度 {min(lengths)}~{max(lengths)}")
    for index, chunk in enumerate(chunks, start=1):
        print(f"  [{index}] {len(chunk)} 字")
        for turn in split_turns(chunk):
            print(f"      {format_turn(turn)}")


def main() -> None:
    text = TEXT_FILE.read_text(encoding="utf-8").strip()
    print(f"原文 {len(text)} 字")

    show(f"按字数切 chunk_size={LENGTH_CHUNK_SIZE}", split_by_length(text))

    embeddings = build_embeddings()
    results = sweep_amounts(text, embeddings)
    amount, chunks = pick_amount(results)

    print(
        f"\n【自动挑阈值】从高往低试，第一个「最长块 ≤ {MAX_CHUNK_LENGTH} 字」的胜出"
        "（percentile 越大切得越少）"
    )
    for candidate, candidate_chunks, longest in results:
        mark = "   <- 选中" if candidate == amount else ""
        print(f"  amount={candidate:>2} -> {len(candidate_chunks):>2} 片，最长 {longest:>4} 字{mark}")

    show(f"按语义切 amount={amount}（块长限制 {MIN_CHUNK_LENGTH}~{MAX_CHUNK_LENGTH} 字）", chunks)


if __name__ == "__main__":
    main()
