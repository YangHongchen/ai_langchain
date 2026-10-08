"""LCEL 最小示例。

LCEL 就一句话：用 | 把 Runnable 串起来，左边吐出来的就是右边的输入。
模型、提示模板、解析器、普通函数都是 Runnable，所以能随便拼；
链一旦拼好，invoke / batch / stream 也一并获得。

跑法：python 006.LCEL.py
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.utils import build_model

# 0、构造模型实例
model = build_model()
parser = StrOutputParser()

# 1、提示词
# 中文注释：system 只负责给模型定角色、立规矩；用户真正要问的事放 human。
# 两边都写同一句话，模型得自己猜以哪个为准，白白添乱。
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的天气助手，回答控制在 30 个字以内。"),
    ("human", "帮我查询{city}的天气"),
])

# 2、构建链（简单）直接拼提示词、模型、输出解析器， 用 | 链接组件，就形成了一个 Runnable 链。
# 中文注释：每一步的类型变化如下，排查问题时按这个顺序看就知道断在哪一环：
#     {"city": "北京"}  ->  PromptValue  ->  AIMessage  ->  str
#       invoke 传进来的     提示模板渲染完    模型返回的对象   解析器取出的文本
chain = prompt | model | parser

# 3、调用执行
result = chain.invoke({"city": "北京"})
print(result)
