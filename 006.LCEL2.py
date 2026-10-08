"""LCEL 最小示例——翻译链。

链的结构：prompt | model | parser
    输入     {"input": "原文", "language": "目标语言"}
    中间输出  模型返回的 AIMessage（一条消息对象，不是字符串）
    最终输出  目标语言的字符串（StrOutputParser 负责取出纯文本）

想换成「英文翻译成中文」，把 invoke 里的 input 和 language 改掉即可：
    chain.invoke({"input": "Hello, I am a student.", "language": "中文"})

跑法：python 006.LCEL2.py
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.utils import build_model

deepseek = build_model()
parser = StrOutputParser()

prompt = ChatPromptTemplate([
    ("system", "你是一个专业的翻译助手，回答控制在 30 个字以内。"),
    ("human", "请将用户输入内容：[{input}]，翻译成{language}"),
])

chain = prompt | deepseek | parser

result = chain.invoke({"language": "法语", "input": "你好，我是一个学生。"})
print(result)