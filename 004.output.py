

    # ---------- 输出解析器（Output Parser）为什么存在 ----------
    #
    # 【为什么存在】模型返回的永远是一个 AIMessage 对象，而不是字符串。
    #   对象里除了 .content 这段文本，还挂着 id、token 用量、工具调用等一堆元数据。
    #   业务代码往往只想要那段文本（存库、拼下一条提示、返回给前端），
    #   如果每处都写 result["messages"][-1].content，既啰嗦，又把业务逻辑
    #   和「消息结构」死死绑在一起——哪天外层多包一层状态字典，全得改。
    #   解析器的职责就一句话：把「模型的输出」翻译成「程序能直接用的数据」。
    #
    # 【有什么作用】两层价值：
    #   1. 类型转换：AIMessage -> str / dict / Pydantic 对象
    #   2. 格式兜底：模型特别爱画蛇添足，比如
    #          "好的，结果如下：\n```json\n{\"name\": \"张三\"}\n```"
    #      解析器负责把代码块、客套话这些壳剥掉，只留 JSON；
    #      解析失败时抛明确异常，而不是让你拿着脏字符串去 json.loads 撞运气。
    #
    # 【最值钱的一点】解析器本身也是一个 Runnable，所以能直接挂在 LCEL 管道末尾，
    #   让整条链的输入输出从「消息对象」变成「干净字符串」：
    #       chain = prompt | model | StrOutputParser()
    #       chain.invoke(...)   # -> str，而不是 AIMessage
    #   手动取 .content 做不到这一点：它没法参与管道组合，也就没法被复用、
    #   被 batch、被 stream。
    #
    # 【常见几种】StrOutputParser（取纯文本，最常用）
    #            JsonOutputParser（解析成 dict，还能把 schema 自动塞进提示词）
    #            PydanticOutputParser（解析成带类型校验的 Pydantic 对象）
    #
    # 【1.0 的取舍】要结构化数据时，现在更推荐 model.with_structured_output(模型类)：
    #   它内置进 agent 主循环，省掉额外一次 LLM 调用。但纯文本和流式场景下，
    #   StrOutputParser 依然是最顺手的收尾。
    #
    # 【用法要点】只接受「单条消息对象」或「字符串」两种输入。
    #   传整个 result 字典进去，它会把 dict 当成文本包成 Generation(text=<dict>)，
    #   于是 pydantic 直接报 ValidationError: Input should be a valid string。
    #   正确做法是先取出单条消息，再交给解析器。

from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain_core.output_parsers import StrOutputParser

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
        context={"user_role": "beginner"},
    )

    # 中文注释：整个 result 是一个状态字典，键是 messages，值是完整的消息列表
    # （含 HumanMessage、AIMessage，每条还带 token 统计等元数据）。
    # 所以 print(result) 出来会很长，真正要的答案在最后一条消息里。
    print("返回类型:", type(result).__name__, "| 顶层键:", list(result))
    last_message = result["messages"][-1]
    print("最后一条消息:", result)
    print('-' * 100)

    str_parser = StrOutputParser()
    str_result = str_parser.invoke(last_message)

    print("StrOutputParser:", str_result)
    print('-' * 100)