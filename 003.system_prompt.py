"""系统提示词（System Prompt）的两种写法。

同一个 agent，两种传提示词的方式：

1. 静态系统提示——提示词写死，构造 agent 时通过 system_prompt 传入：

       agent_v1 = create_agent(
           model=build_model(),
           tools=[],
           system_prompt="你是一位简洁、友好的 AI 助手。",
       )

2. 动态系统提示（Dynamic System Prompt）——提示词按运行时上下文（比如用户角色）
   临时生成，需要借助 @dynamic_prompt 中间件。即本文件下半部分的 agent_v2。

跑法：python system_prompt.py
"""

from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest

from common.utils import build_model


# ---------- 1. 先声明运行时上下文的格式 ----------
# 中文注释：必须用 TypedDict。下面中间件里是 request.runtime.context.get(...) 这样取值，
# .get() 是 dict 的方法，普通 class 没有；官方示例同样用 TypedDict。
class Context(TypedDict):
    """agent 每次调用时可以传入的运行时上下文。"""

    user_role: str  # "expert" | "beginner"，不传时按普通用户处理


# ---------- 2. 用 @dynamic_prompt 声明动态系统提示 ----------
# 中文注释：这个装饰器会把函数包装成一个中间件；中间件在每次调用模型之前执行，
# 返回的字符串就是本次请求要用的 system prompt。
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成不同的系统提示。

    中文注释：@dynamic_prompt 调用被装饰函数时传入的是 ModelRequest，
    所以形参必须叫 request —— 写成别的名字，函数体里的 request 就成了未定义的全局名。

    Args:
        request: 本次模型请求的信息，运行时上下文挂在 request.runtime.context 上。

    Returns:
        本次请求要使用的系统提示词。
    """

    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是友好的 AI 助手。"

    if user_role == "expert":
        return f"{base_prompt}，请提供专业级别的指导，并协助用户完成任务。"
    if user_role == "beginner":
        return f"{base_prompt}，简单介绍下概念，避免行话或专业术语。"
    return base_prompt


# ---------- 3. 把中间件挂到 agent 上 ----------
agent_v2 = create_agent(
    model=build_model(),
    tools=[],
    middleware=[user_role_prompt],
    # 声明上下文格式：invoke 时传的 context= 需要符合这个结构
    context_schema=Context,
)


if __name__ == "__main__":
    # 中文注释：invoke 的输入必须是 {"messages": [...]}，不能直接给字符串；
    # 运行时上下文则通过 context= 传入，最终在中间件里由 request.runtime.context 取到。
    result = agent_v2.invoke(
        {"messages": ["解释下什么是机器学习"]},
        context={"user_role": "expert"},
    )
    print(result["messages"][-1].text)
