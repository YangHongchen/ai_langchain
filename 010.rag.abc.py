"""为什么需要 RAG（检索增强生成）？

本文件不实现 RAG，只用一次对照实验说明它解决什么问题：

    第 1 步  直接问模型                 → 它不知道（或者编一个）
    第 2 步  把资料原文塞进提示词再问     → 答对了
    第 3 步  那为什么不每次全塞？         → 塞不下、太贵、反而更差
    第 4 步  所以：先检索相关片段再塞      → 这就是 RAG

跑法：python 010.rag.abc.py
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.utils import build_model

# 一份编造的「公司内部文档」。模型训练数据里绝对没有这些内容 ——
# 这正是私有知识的典型场景：它真实存在，但模型没见过，所以答不出来。
DOCUMENT = """《星辰科技员工手册（2024 版）》节选

第 7 条 年假制度：正式员工入职满一年享 12 天年假，满三年后增至 18 天。
第 12 条 远程办公：每周最多申请 2 天远程办公，需提前一天在系统里提交。
第 19 条 交通补贴：每人每月 800 元，超出部分需部门总监审批。
"""

QUESTION = "星辰科技的员工每年有几天年假？"

model = build_model()
parser = StrOutputParser()


# ---------- 第 1 步：直接问，不给资料 ----------
prompt_plain = ChatPromptTemplate.from_messages([("human", "{question}")])
answer_plain = (prompt_plain | model | parser).invoke({"question": QUESTION})


# ---------- 第 2 步：把资料原文一起塞进提示词 ----------
# 中文注释：只多了 system 里那一段资料，模型、问法、参数全都没变。
# 这一步是关键证据：说明模型不是「笨」，而是「没见过这份资料」。
prompt_grounded = ChatPromptTemplate.from_messages(
    [
        ("system", "只根据下面的资料回答；资料里没有就说不知道。\n\n资料：\n{document}"),
        ("human", "{question}"),
    ]
)
answer_grounded = (prompt_grounded | model | parser).invoke(
    {"question": QUESTION, "document": DOCUMENT}
)


print("=" * 70)
print(f"问题：{QUESTION}")
print("=" * 70)

print("\n第 1 步 · 直接问模型（不给资料）")
print(f"  答：{answer_plain.strip()}")

print("\n第 2 步 · 把资料塞进提示词再问（同一个模型、同一个问法）")
print(f"  答：{answer_grounded.strip()}")

# ---------- 第 3 步：那为什么不每次全塞？ ----------
print("\n第 3 步 · 那为什么不干脆每次都把文档全塞进去？")
print(f"  这份文档才 {len(DOCUMENT)} 个字，塞进去当然没问题。")
print("  但真实场景不是一份文档，而是一个知识库：几百上千份、几十万字。于是三个问题同时出现：")
print("    · 装不下 —— 模型一次能看的字数（上下文窗口）有硬上限")
print("    · 太贵   —— 输入按 token 计费，全塞等于每次提问都为整个知识库付一遍钱")
print("    · 反而差 —— 90% 的无关内容会稀释注意力，模型更容易抓错重点")

# ---------- 第 4 步：所以需要 RAG ----------
print("\n第 4 步 · 把「全塞」换成「先挑出最相关的几段，再塞」")
print(
    """
    用户提问
       ↓
    [检索] 从知识库里找出最相关的 N 个片段     ← RAG 里的 R (Retrieval)
       ↓
    [增强] 把这几个片段拼进提示词              ← A (Augmented)
       ↓
    [生成] 模型只看这几个片段来回答            ← G (Generation)
"""
)
print("  检索把「几十万字」压成「几百字」，上面三个问题一起消失。")
print("  至于「怎么找出最相关」——切分、向量化、相似度检索，那是下一步的内容。")
