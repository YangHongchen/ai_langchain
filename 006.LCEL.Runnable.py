"""LCEL 链：RunnableLambda 与 itemgetter。

itemgetter 按 key 从输入 dict 里取值：
    d = {"foo": "123", "bar": "456"}
    itemgetter("foo")(d)          # -> "123"

两个要点：
1. @chain（来自 langchain_core.runnables）把普通函数包装成 RunnableLambda，
   包装完才能参与 | 运算。注意别写成 itertools.chain —— 那是迭代器工具，同名但完全无关。
2. dict 放在 | 的左边会被自动转成 RunnableParallel：
   多个分支并行执行，结果合并成一个 dict。

跑法：python 006.LCEL.Runnable.py
"""
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, chain

from common.utils import build_model

template = ChatPromptTemplate.from_template("{a} * {b} 结果是多少？")
deepseek = build_model()
parser = StrOutputParser()

# 定义一个简单的 Lambda 函数，用于计算字符串的长度
def length(s: str) -> int:
    return len(s)

# 将两个字符串的长度相乘
def mul(t1: str, t2: str) -> int:
    return len(t1) * len(t2)

# @chain 是 RunnableLambda 的装饰器，用于将 Lambda 函数转换为链
# 中文注释：入参 d 是上一环拼出来的 {"t1": ..., "t2": ...}，
# 所以这里要按 t1 / t2 取，别按原始的 name / sex 取（那是上游的字段名）。
@chain
def mul_length(d):
    return mul(d["t1"], d["t2"])

chain1 = template | deepseek

chain2 = (
    # 中文注释：显式写 RunnableParallel，而不是丢一个 dict 让框架隐式转换。
    # 两者运行结果完全一样，但显式写法类型是清楚的（Runnable | Runnable），
    # 不会被 IDE / 类型检查器报「dict 不支持 | 运算」，也一眼能看出这里是并行。
    RunnableParallel(
        # a 分支：取出 name，算出它的长度。6
        # 注意 itemgetter(...) 本身不是 Runnable，直接放在 | 左边同样会被警告
        # （operator.itemgetter 没有定义 __or__），所以先用 RunnableLambda 包一层。
        a=RunnableLambda(itemgetter("name")) | RunnableLambda(length),
        # b 分支：先把 name / sex 两个字段并行抽成 {"t1": ..., "t2": ...}，
        # 再交给 mul_length 算长度之积。6 * 4 = 24
        b=RunnableParallel(
            t1=RunnableLambda(itemgetter("name")),
            t2=RunnableLambda(itemgetter("sex")),
        )
        | mul_length,
    )
    | chain1
    | StrOutputParser()
)



print(chain2.invoke({"name": "wangwu", "sex": "male"}))
print("-" * 100)
