"""RAG
第三步（补充）：相似度计算 —— 向量之间怎么比「谁更像谁」。

【三种度量，一张表说清】
             公式                 取值范围      越大越像？
    ------------------------------------------------------------
    cosine   a·b / (|a|·|b|)      -1 ~ 1        是
    ip       a·b                  不定          是
    l2       |a - b|              0 ~ 无穷      否（越小越像）

  cosine（余弦相似度）：只看【方向】，不管长短。一条 5 个字、一条 50 个字，
        只要讲的是同一件事，余弦照样接近 1 —— 所以文本检索默认用它。
        注意别念错：cos 本身是「相似度」，范围 -1~1；
        「1 - cos」才是余弦【距离】，范围 0~2，越小越像。
  ip（点积）：方向 × 长度一起算。向量越长分越高，所以没归一化的向量不能直接比。
  l2（欧氏距离）：两点之间的直线距离。它是【越小越像】，排序方向跟前两个相反，写代码时最容易在这里搞反。

【实测结论：归一化之后，三者排名完全一样】
  fastembed 吐出来的向量【已经做过归一化】—— 每个向量的模长都是 1.000000
  （本文件会把它打印出来）。向量长度都是 1 的时候：

      cos = a·b                 分母 |a|·|b| = 1×1 = 1，余弦直接退化成点积
      |a-b|² = 2 - 2·(a·b)      欧氏距离和点积是严格的换算关系

  所以三种算法排出来的名次一模一样，只是数值范围不同（本文件最后会验证）。
  但【别把这当成普遍规律】—— 换个不做归一化的 embedding 模型，选错度量是真的会把
  结果排错。真正通行的做法是：统一先用 cosine，最稳。

【这几个名字你在哪儿见过】
  'cosine' / 'l2' / 'ip' 也正是向量库配置距离算法的取值 —— 建集合时要选一个，
  选完就固定，之后每次检索都按它排序。后面讲向量库时会再遇到。

跑法：python 021.rag.embedding.similarity_calc.py
"""
import common.setup  # noqa: F401
import numpy as np

from common.utils import build_embeddings

QUERY = "公司年假有几天"

# 中文注释：前两条都跟年假有关，第三条完全跑题 —— 用来验算三种度量排出的名次是否一致。
DOCUMENTS = [
    "年假：入职满一年享 12 天",
    "休假制度：满一年 12 天，满三年 18 天",
    "公司门口新开了一家川菜馆，水煮鱼很好吃",
    "公司门口新开了一家湘菜馆，水煮鱼很好吃",
]

def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """余弦相似度：两个向量夹角的余弦值，1 = 同向（像），0 = 垂直（无关）。"""
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def inner_product(left: np.ndarray, right: np.ndarray) -> float:
    """点积：对应位置相乘再求和。"""
    return float(left @ right)


def euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    """欧氏距离：两点间的直线距离。注意它越小越像，排序方向和另外两个相反。"""
    return float(np.linalg.norm(left - right))


def rank_order(scores: list[float], bigger_is_better: bool) -> list[int]:
    """把分数转成名次列表（最像的排最前），返回的是候选文档的下标顺序。"""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=bigger_is_better)


def main() -> None:
    embeddings = build_embeddings()
    query_vector = np.array(embeddings.embed_query(QUERY))
    document_vectors = [np.array(vector) for vector in embeddings.embed_documents(DOCUMENTS)]

    print(f"查询：{QUERY}")
    # 中文注释：模长是不是 1，决定了三种度量能不能互相替代 —— 所以这里先把它亮出来。
    print(f"查询向量模长：{np.linalg.norm(query_vector):.6f}（1.0 说明已经归一化）")

    cosines, products, distances = [], [], []
    print("\n【三种度量算出来的分数】")
    for index, document_vector in enumerate(document_vectors):
        cosine = cosine_similarity(query_vector, document_vector)
        product = inner_product(query_vector, document_vector)
        distance = euclidean_distance(query_vector, document_vector)
        cosines.append(cosine)
        products.append(product)
        distances.append(distance)
        print(f"  [{index}] {DOCUMENTS[index]}")
        print(f"       余弦 {cosine:+.4f}   点积 {product:+.4f}   欧氏距离 {distance:.4f}")

    print("\n【排名对比】0 = 最像，数组里的数字是候选文档下标")
    print(f"  余弦相似度（越大越像）-> {rank_order(cosines, bigger_is_better=True)}")
    print(f"  点积      （越大越像）-> {rank_order(products, bigger_is_better=True)}")
    print(f"  欧氏距离  （越小越像）-> {rank_order(distances, bigger_is_better=False)}")

if __name__ == "__main__":
    main()
