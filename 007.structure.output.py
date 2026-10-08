"""结构化输出（Structured Output）——让模型返回对象，而不是一段文本。

【解决什么问题】
普通调用拿回来的是字符串，你还得自己从里面抠字段：
    "好的，提取结果如下：姓名 jeff，邮箱 jeff@163.com ……"
结构化输出让模型直接给你一个 Pydantic 对象，字段和类型都校验过，
可以直接存库、传给下游 API，不用写正则去解析。

【怎么用】
给 create_agent 传 response_format，两种策略：
    ToolStrategy(模型类)     靠工具调用实现，兼容性最好，支持 tool calling 的模型都能用
    ProviderStrategy(模型类) 用供应商原生的结构化输出能力，更稳，但只有部分模型支持
直接传模型类也行，LangChain 会按模型能力自动挑一种。

【返回值在哪】
agent 返回的状态字典多了一个键 structured_response，注意它不在 messages 里：
    result["structured_response"]  ->  ContactInfo 实例

跑法：python 007.structure.output.py
"""
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from common.utils import build_model

class ContactInfo(BaseModel):
    """从一段文字里抽取出的联系人信息。

    中文注释：字段的 description 不是给你看的，是给模型看的——
    它会作为工具参数的说明一起发给模型，直接影响抽取准确率，值得认真写。
    """

    name: str = Field(description="姓名")
    email: str = Field(description="邮箱地址")
    phone: str = Field(description="手机号，只保留数字")

# ToolStrategy：把「输出格式」包装成一个工具交给模型，
# 模型通过「调用这个工具」来完成输出。这就是它能兼容各家模型的原因。
agent = create_agent(
    model=build_model(),
    tools=[],
    response_format=ToolStrategy(ContactInfo),
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "从 jeff, jeff@163.com, 19989897654 中提取联系人信息"}
    ]
})

# ---------- 取答案 ----------
# 中文注释：答案是 result["structured_response"]，一个 ContactInfo 实例。
# 不要再接 StrOutputParser —— 它是给「单条消息」用的，而 agent 返回的是整个状态字典，
# 喂进去会报 ValidationError: Input should be a valid string。
info: ContactInfo = result["structured_response"]

print("对象     :", result)
# print("字段访问 :", f"{info.name} / {info.email} / {info.phone}")
# # 结构化输出真正的价值在这两行：拿到就能直接用，不需要任何解析
# print("转 dict  :", info.model_dump())
# print("转 JSON  :", info.model_dump_json())

# ---------- 顺带看看 messages 里发生了什么 ----------
# 中文注释：消息列表末尾会多一条 ToolMessage，内容是 "Returning structured response: ..."。
# 这正是 ToolStrategy 的工作痕迹：模型并没有「直接返回对象」，
# 它只是调用了 LangChain 临时挂上去的那个工具，拿到工具调用后框架再帮你转成对象。
# print("\nmessages :", [type(message).__name__ for message in result["messages"]])
# print("最后一条 :", repr(result["messages"][-1].text)[:70])
