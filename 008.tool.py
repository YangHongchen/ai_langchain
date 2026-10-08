"""LangChain 工具（Tool）用法。

【一句话】工具就是「给模型挂一批它能自己调用的函数」。
模型看到的是每个工具的 name、description 和参数 schema，看不到函数体。
所以 docstring 不是注释，是提示词 —— 要写清楚「什么场景下用我」。

【两种用法，本文件都演示】
1. 直接当函数调用：search_database.invoke({"query": "..."})  —— 你指定，不经过模型
2. 挂给 agent：模型自己决定调不调、调哪个、调几次、什么时候停

跑法：python 008.tool.py
"""
from langchain.agents import create_agent
from langchain_core.tools import tool

from common.utils import build_model


# ---------- 1. 定义工具 ----------
# 中文注释：三个工具的职责必须互不重叠，否则模型会选错。
# docstring 要写「什么时候用它」，而不是「它是什么」—— 这是影响选择准确率最关键的东西。
# 另外 @tool 要求必须有 docstring，或者显式传 description，否则直接报错。

@tool
def search_database(query: str, limit: int = 10) -> str:
    """查询公司内部数据库里的结构化数据，例如用户、订单、库存记录。"""
    return f"查到 {limit} 条与「{query}」相关的记录。"


@tool("web_search")
def web_search(query: str) -> str:
    """联网搜索实时信息，比如天气、新闻、股价。"""
    return f"「{query}」的联网搜索结果。"


@tool("calculator", description="执行数学计算，解决算术问题。")
def calc(expression: str) -> str:
    """计算数学表达式，支持 + - * / 和括号。"""
    # 中文注释：演示够用。真实项目里 expression 是模型生成的，
    # 直接 eval 等于允许它在你机器上执行任意代码，要换成白名单求值。
    return str(eval(expression))


# ---------- 2. 直接调用：工具就是个可调用对象 ----------
# 中文注释：入参是 dict，key 就是函数签名里的参数名。
# 这条路完全不经过模型 —— 流程固定时这么用，可控又省钱。
print("直接调用 :", search_database.invoke({"query": "订单", "limit": 5}))

# 模型眼里的工具就长这样，它做选择时只看这几个字段
print("工具描述 :", f"{web_search.name} -> {web_search.description}")

# ---------- 3. 交给 agent：模型自己决定用哪个 ----------
agent = create_agent(build_model(), tools=[search_database, web_search, calc])

for question in ("长沙明天天气怎么样， 优先查询本地已同步的天气？", "数据库里有多少用户？", "23 * 47 等于多少？"):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # 中文注释：从消息里翻出模型实际发起的工具调用，看它选了什么。
    # 模型在一个循环里可能调多次、调多个，所以这里收集成列表。
    picked = [
        call["name"]
        for message in result["messages"]
        if getattr(message, "tool_calls", None)
        for call in message.tool_calls
    ]

    print(f"\n问：{question}")
    print(f"  模型选了：{picked or '（没调工具）'}")
    print(f"  最终回答：{result['messages'][-1].text}")
