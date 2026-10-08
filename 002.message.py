"""LangChain 消息类型演示。

在 LangChain 中，消息（Message）是模型的基本上下文单元，代表模型的输入和输出。
每条消息包含：

- 角色（role）：消息的发送者，如 system、human、ai、tool
- 内容（content）：消息的具体文本内容
- 其他字段：随类型而异，比如 ai 消息的 tool_calls、tool 消息的 tool_call_id
"""

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# 复用 common/utils.py 里已经处理好的模型构造（含自定义的 DEEPSEEK_* 变量名），
# 避免各处都写一遍 API Key 和 base_url。
from common.utils import build_model

# ---------- 示例一：显式消息类。这里排成一次完整的工具调用往返 ----------
# 顺序不能乱：system 定行为 -> human 提问 -> ai 申请调用工具 -> tool 回填结果。
#
# 注：老教程里的 FunctionMessage 是 OpenAI function calling 时代的产物，
# 现在已废弃，统一用 ToolMessage 代替，所以这里不再导入它。
messages = [
    SystemMessage(content="你是一个专业的翻译。"),
    HumanMessage(content="把『你好』翻译成英文。"),
    # 模型决定调用工具时，content 通常是空的，关键信息在 tool_calls 里。
    # 注意 tool_calls 属于 AIMessage，不属于 ToolMessage —— 这是最容易写反的地方。
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "translate",
                "args": {"text": "你好", "target": "en"},
                # id 是配对凭据：下面回填结果时要拿它来对上号
                "id": "call_demo_1",
            }
        ],
    ),
    # 回填工具结果：tool_call_id 必须等于上面那次申请的 id。
    # ToolMessage 的必填字段是 content + tool_call_id，没有 tool_calls。
    ToolMessage(content="Hello", tool_call_id="call_demo_1"),
]

# ---------- 示例二：字典写法，一段普通多轮对话 ----------
# role 只认 system / user / assistant / tool（human、ai 是 user、assistant 的别名）。
# "function" 是 OpenAI 早期角色，已经废弃；tool 角色还必须带 tool_call_id。
# 这里最后一条刻意是 user，因为模型要回答的始终是最后那条用户消息，
# 前面的 assistant 消息就是「历史」，也就是多轮记忆的全部秘密。
message2 = [
    {"role": "system", "content": "你是一个专业的翻译。"},
    {"role": "user", "content": "写一句关于春天的七言绝句。"},
    {"role": "assistant", "content": "桃花流水窅然去，别有天地非人间。"},
    {"role": "user", "content": "把这句翻译成英文。"},
]


def main() -> None:
    """把消息列表交给 agent 走一轮。"""

    agent = create_agent(
        # create_agent 的签名是 (model, tools, *, system_prompt=..., debug=...)：
        # 没有 messages 参数，没有 agent_type，也没有 verbose（看细节用 debug）。
        model=build_model(),
        tools=[],
        # debug=True 会把每一步的完整状态打出来，调试很有用，平时可以关掉
        debug=True,
    )

    # 消息列表是 invoke 的入参，不是构造参数；agent 的输入输出都包在 "messages" 里
    result = agent.invoke({"messages": message2})
    print(result["messages"][-1].text)


if __name__ == "__main__":
    # 中文注释：一定要加这个守卫。原来文件末尾直接写 exit()，
    # 那样任何 import message 都会当场终止 Python 进程，属于隐藏的炸弹。
    main()
