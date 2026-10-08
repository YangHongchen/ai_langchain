"""RAG
第四步：向量库（Chroma）—— 把向量存起来，并且能快速找出最像的几条。

【它解决什么问题】
向量算出来之后，如果只放在一个 python list 里，检索就是一条条比对 ——
10 万条文档 = 10 万次距离计算，慢到没法用。Chroma 干两件事：
    存：原文 + 向量 + metadata 一起落盘，进程关了还在
    查：用 HNSW 索引近似地找出最像的 top-k，比全量比对快几个数量级

【最小用法就三行】
    store = Chroma.from_texts(片段, embedding=模型, persist_directory="chroma_db")
    store.similarity_search(问题, k=3)                        -> 最像的几条
    store.similarity_search_with_relevance_scores(问题, k=3)   -> 再带上 0~1 的相似度

【距离算法：建库时定死，事后改不了】
Chroma 支持 cosine / l2 / ip 三种，在创建 collection 时指定，建完就改不了 ——
因为 HNSW 索引是按那种距离的数学性质构建的，想换只能删了重建。
好在本项目的向量已经归一化，三种算法排名完全一样（本文件会验证），选哪个都行。

【两个会踩的坑】
  1. 建库和检索必须用【同一个 embedding 模型】。换了模型，两边的向量不在同一个空间里，检索结果就是乱的 —— 而且它不报错，只是悄悄变差。
  2. Chroma【不做去重】。这个 demo 每次运行都把同一批文档插一遍，不清理的话库会越堆越大，检索结果里全是重复片段。所以建库前先 reset_collection()。

跑法：python 022.rag.vectorstore.chroma.py
"""
from pathlib import Path

import common.setup  # noqa: F401
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.utils import build_embeddings

TEXT_FILE = Path("assets/sample_call.txt")
PERSIST_DIR = Path("chroma_db")
COLLECTION_PREFIX = "call_transcript"

# 中文注释：三种距离算法各建一个 collection —— 名字不能重复，否则后建的会覆盖前面的配置。
SPACES = ["cosine", "l2", "ip"]

CHUNK_SIZE = 200

# 中文注释：中文标点分离器。切分本身 016~020 已经讲透了，这里照抄一份，一行带过。
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

QUERIES = ["上门要不要收费", "IPTV 卡顿跟宽带是同一个问题吗"]

TOP_K = 3
PREVIEW_LIMIT = 44


def collection_name(space: str) -> str:
    """每种距离算法用独立的 collection 名字。"""

    return f"{COLLECTION_PREFIX}_{space}"


def split_text(text: str) -> list[str]:
    """把通话记录切成片段，返回 list[str]。"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
        separators=SEPARATORS,
    )
    return splitter.split_text(text)


def build_store(chunks: list[str], embeddings: Embeddings, space: str) -> Chroma:
    """建库：Chroma 会用我们给的 embedding 算好向量并存下来。"""

    store = Chroma(
        collection_name=collection_name(space),
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
        # 中文注释：距离算法从这里传进去。注意是 collection_configuration，
        # 不是老教程里的 collection_metadata={"hnsw:space": ...} —— 新版只从 configuration 读。
        collection_configuration={"hnsw": {"space": space}},
    )

    # 中文注释：见文件头第 2 个坑 —— 每次运行都会重新插入同一批文档，
    # 而 Chroma 不去重，所以先清空这个 collection 再灌。
    store.reset_collection()
    store.add_texts(chunks)

    return store


def preview(document: Document) -> str:
    """截一段正文用于单行预览。"""

    return " ".join(document.page_content.split())[:PREVIEW_LIMIT] + "…"


def search(store: Chroma, query: str, k: int = TOP_K) -> None:
    """给个问题，拿回最像的 k 条。

    中文注释：这里用 similarity_search_with_relevance_scores，它返回的是【相似度】
    0~1、越大越像，读起来比距离直观。想拿原始距离就用 similarity_search_with_score。
    """

    print(f"\n查询：{query}")
    for rank, (document, score) in enumerate(
        store.similarity_search_with_relevance_scores(query, k=k), start=1
    ):
        print(f"  [{rank}] 相似度 {score:.4f}  {preview(document)}")


def compare_spaces(stores: dict[str, Chroma], query: str, k: int = TOP_K) -> None:
    """三种距离算法各查一遍：分数不一样，但排名一样吗？"""

    print(f"\n查询：{query}（每行是三条命中的相似度）")
    orders = {}
    for space, store in stores.items():
        hits = store.similarity_search_with_relevance_scores(query, k=k)
        orders[space] = [document.page_content for document, _ in hits]
        scores = "   ".join(f"{score:.4f}" for _, score in hits)
        print(f"  space={space:<7} {scores}")

    # 中文注释：把三条命中拼成元组去重 —— 只剩一种，说明三种算法给出的排名完全一样。
    distinct = len({tuple(order) for order in orders.values()})
    print(f"  -> 三种算法的排名{'完全一致' if distinct == 1 else f'有 {distinct} 种不同'}")


def main() -> None:
    text = TEXT_FILE.read_text(encoding="utf-8").strip()
    chunks = split_text(text)
    embeddings = build_embeddings()
    print(f"原文 {len(text)} 字 -> 切成 {len(chunks)} 片")

    stores = {space: build_store(chunks, embeddings, space) for space in SPACES}
    print(f"已写入 Chroma：{PERSIST_DIR}（{len(SPACES)} 个 collection，各 {len(chunks)} 条）")

    print("\n========== 一、检索：给个问题，拿回最像的几条 ==========")
    for query in QUERIES:
        search(stores["cosine"], query)

    print("\n========== 二、距离算法能换，但影响不到召回 ==========")
    compare_spaces(stores, QUERIES[0])

    # ---------- 三、验证「真的落盘了」 ----------
    # 中文注释：重新 new 一个 Chroma 指向同一个目录，不传任何文本，看能不能查到 ——
    # 能查到就说明向量真的存在磁盘上，不是活在这个进程的内存里。
    print("\n========== 三、数据真的在磁盘上 ==========")
    reopened = Chroma(
        collection_name=collection_name("cosine"),
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    print(f"重新打开磁盘上的库，还是 {len(reopened.get()['ids'])} 条")


if __name__ == "__main__":
    main()
