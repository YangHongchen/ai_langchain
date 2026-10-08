"""RAG
第三步：向量化（Embedding）—— 把文字变成一串数字，让机器能算「意思有多近」。

【为什么需要这一步】
切分只是把文章切成片，片里还是文字。而文字没法做相似度计算 ——「年假」和「休假」在计算机眼里只是四个不同的汉字，毫无关系。
embedding 模型把一段文字压成一串固定长度的浮点数（就叫「向量」），
意思越接近的两段文字，它们的向量在多维空间里挨得越近。

    年假：入职满一年享 12 天   ->  [0.023, -0.041, ...]   ┐ 夹角小
    休假：满一年 12 天          ->  [0.019, -0.038, ...]   ┘
    楼下新开了家川菜馆         ->  [-0.07,  0.052, ...]   ← 岔开了

【两个方法，别用混】
    embed_documents(texts)   一次转一批，建库/切分用
    embed_query(text)        转一条查询，检索时用
为什么要分开？因为有些模型（BGE 系列就是）给「查询」和「文档」补的前缀不一样，
用错了会让检索效果偷偷变差，还不报错。

【怎么衡量「近」—— 余弦相似度】
两个向量夹角的余弦值：1 = 完全同向（意思一样），0 = 完全没关系。
把它反过来就是「距离」：距离 = 1 - 相似度。
这正是 020 里 SemanticChunker 判断该不该切刀用的那个数。

【本项目的模型选择】
DeepSeek 只提供对话接口，没有 embeddings 接口（实测 /embeddings 返回 404），
所以向量化走本地 ONNX 模型：BAAI/bge-small-zh-v1.5（512 维，约 91 MB，权重放在项目根的 models/ 下，运行时零网络请求）。
统一入口是 common.utils.build_embeddings()，和 build_model() 一样只改一处。

跑法：python 021.rag.embedding.ABC.py
"""

import common.setup  # noqa: F401
from langchain_core.embeddings import Embeddings

from common.utils import build_embeddings

# 中文注释：三组用来演示「语义远近」的句子 —— 前两句讲同一件事，第三句换了话题，
# 正好看相似度怎么分化。
SENTENCES = [
    "年假：入职满一年享 12 天",
    "休假制度：满一年 12 天，满三年 18 天",
    "公司门口新开了一家川菜馆，水煮鱼很好吃",
    "公司门口新开了一家湘菜馆，水煮鱼很好吃",
]

# 中文注释：向量有 512 个数，全打出来没法看，只取前几个。
PREVIEW_SIZE = 5

def show_vector(name: str, vector: list[float]) -> None:
    """打印向量的维度和前几个数 —— 用来让人对「向量长什么样」有个直观印象。"""

    head = ", ".join(f"{value:+.4f}" for value in vector[:PREVIEW_SIZE])
    print(f"{name}：{len(vector)} 维，前 {PREVIEW_SIZE} 个值 [{head}, …]")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """算两个向量的余弦相似度，返回 0~1（1 = 意思一样，0 = 没关系）。

    中文注释：这里手写一遍是为了让你看清公式。实际项目直接用现成的就行，
    比如 langchain_community.utils.math.cosine_similarity，不用自己造轮子。
    """

    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    return dot / (norm_left * norm_right)


def main() -> None:
    embeddings: Embeddings = build_embeddings()

    # ---------- 一、一条文本 -> 一个向量 ----------
    text = "这个是一个测试文本"
    print(f"原文：{text}")
    show_vector("嵌入向量", embeddings.embed_query(text))

    # ---------- 二、一批文本 -> 一批向量 ----------
    print(f"\n【三组句子】")
    for index, sentence in enumerate(SENTENCES):
        print(f"  [{index}] {sentence}")

    vectors = embeddings.embed_documents(SENTENCES)
    print(f"\n一次转 {len(SENTENCES)} 条，每条都是 {len(vectors[0])} 维")

    # ---------- 三、意思越近，向量越近 ----------
    print("\n【两两相似度】")
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            similarity = cosine_similarity(vectors[i], vectors[j])
            print(f"  [{i}]-[{j}]  相似度 {similarity:.4f}   距离 {1 - similarity:.4f}")


if __name__ == "__main__":
    main()
