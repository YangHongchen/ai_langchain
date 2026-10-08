"""LangChain Hello World."""

import os
import sys
from dataclasses import dataclass
from importlib.util import find_spec

import langchain
import langchain_core


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from common.utils import build_model



@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""

    # 中文注释：真实项目这里换成 HTTP 请求；演示时固定返回，
    # 好让注意力留在「消息怎么往返」上，而不是数据从哪来。
    return f"{city}今天晴，26℃，微风。"


def main() -> None:
    """LangChain Hello World。"""

    model = build_model()
    # chain = prompt | model | StrOutputParser()
    # result = chain.invoke(
    #     {"message": "给我讲一个笑话"}
    # )
    # print(result)

    # create_agent 的签名是 create_agent(model, tools, *, system_prompt=..., debug=...)，
    # 和上面 LCEL 链的写法有两处不同，照老例子抄最容易踩：
    # 1) 没有 prompt 参数，系统提示叫 system_prompt，且只接受 str / SystemMessage，
    #    ChatPromptTemplate 不能直接传进来（要动态改提示得用 middleware）
    # 2) 没有 verbose 参数，想看每一步执行细节用 debug=True
    ds = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一位简洁、友好的 AI 助手。",
        debug=True,
    )

    # agent 的入参和出参都是消息列表而不是单个字符串，所以要包一层 "messages"；
    # 真正要的答案在最后一条 AI 消息的 content 里
    result = ds.invoke({"messages": [{"role": "user", "content": "给我讲一个笑话"}]})
    print(result["messages"][-1].content)


def demo_messages() -> None:
    """演示 LangChain 的消息（Message）用法。

    中文注释：消息是模型调用真正的输入输出单位——LCEL 里的模板、agent 里的历史、
    工具调用的往返，最终都被翻译成一条条消息。这里按「怎么写 -> 怎么带历史 ->
    长什么样 -> 怎么流式 -> 工具怎么往返」的顺序过一遍，跑完之后 agent 就不再是黑盒了。
    """

    model = build_model()

    # ---------- 1. 同一段对话的三种写法 ----------
    print("1. 同一段对话的三种写法，模型收到的东西完全一样")

    as_tuple = [
        ("system", "你是一位简洁的助手。"),
        ("human", "用一句话说明消息在 LangChain 里是什么。"),
    ]
    as_dict = [
        {"role": "system", "content": "你是一位简洁的助手。"},
        {"role": "user", "content": "用一句话说明消息在 LangChain 里是什么。"},
    ]
    as_object = [
        SystemMessage("你是一位简洁的助手。"),
        HumanMessage("用一句话说明消息在 LangChain 里是什么。"),
    ]

    for name, messages in (
        ("tuple 简写", as_tuple),
        ("dict", as_dict),
        ("显式消息类", as_object),
    ):
        reply = model.invoke(messages)
        print(f"   [{name}] 返回 {type(reply).__name__}: {reply.text}")

    # ---------- 2. 历史消息就是「记忆」 ----------
    print("\n2. 多轮对话：把历史整段发回去，才叫有记忆")

    history: list[BaseMessage] = [SystemMessage("你是一位简洁的助手。")]
    for question in ("我叫 Bruce，请记住。", "我叫什么？"):
        history.append(HumanMessage(question))
        reply = model.invoke(history)
        # 关键：模型自己的回复也要塞回历史，下一轮它才「看得到」自己说过什么。
        # 所谓 Memory / 多轮记忆，本质就是这一步，没有任何魔法。
        history.append(reply)
        print(f"   问: {question}")
        print(f"   答: {reply.text}")
    print(f"   历史里现在有 {len(history)} 条消息：1 system + 2 human + 2 ai")

    # ---------- 3. 消息对象长什么样 ----------
    print("\n3. 消息对象的关键字段")

    last = history[-1]
    print(f"   类型            : {type(last).__name__}")
    print(f"   .text           : {last.text}")
    print(f"   .id             : {last.id}")
    print(f"   .usage_metadata : {last.usage_metadata}")

    # ---------- 4. 流式 ----------
    print("\n4. 流式：拿到的是 AIMessageChunk，可以用 + 拼回一条完整消息")

    merged = None
    for piece in model.stream([HumanMessage("只用三个字回答：今天天气好吗？")]):
        merged = piece if merged is None else merged + piece
    print(f"   拼接后类型: {type(merged).__name__}")
    print(f"   内容      : {merged.text}")

    # ---------- 5. 工具调用的消息往返 ----------
    print("\n5. 工具调用：模型只负责「申请」，结果必须用 ToolMessage 回填")

    bound = model.bind_tools([get_weather])
    question = HumanMessage("北京今天天气怎么样？")
    ai_message = bound.invoke([question])

    print(f"   模型返回类型: {type(ai_message).__name__}")
    print(f"   .text       : {ai_message.text!r}（决定调工具时这里通常是空的）")
    for call in ai_message.tool_calls:
        print(f"   .tool_calls : {call['name']}({call['args']}) id={call['id']}")

    # tool_call_id 是配对的唯一凭据：一条回复里可能有多个工具调用，
    # 少了它模型就分不清哪个结果对应哪次申请。
    tool_results = [
        ToolMessage(content=get_weather.invoke(call["args"]), tool_call_id=call["id"])
        for call in ai_message.tool_calls
    ]

    final = bound.invoke([question, ai_message, *tool_results])
    print(f"   回填 ToolMessage 之后: {final.text}")


@dataclass(frozen=True)
class ModelProvider:
    """一家模型供应商的接入信息。

    中文注释：把「类名 / pip 包 / 环境变量 / 初始化代码」这四样东西绑在一起，
    以后要加新的供应商，只改 _PROVIDERS 一处即可，清单和可用性自检会自动跟上，
    避免文档、代码、依赖三处各写一遍然后逐渐对不上。
    """

    key: str  # 传给 init_chat_model 的 provider 标识
    label: str  # 展示用的名字
    class_name: str  # 初始化时要用的类
    package: str  # 需要 pip install 的包名
    env_vars: tuple[str, ...]  # 必需的环境变量，空元组表示不需要
    snippet: str  # 初始化代码片段
    note: str = ""  # 容易踩的坑


# 各家接入方式。顺序按国内项目常用度排，不按厂商字母序。
_PROVIDERS: tuple[ModelProvider, ...] = (
    ModelProvider(
        key="openai",
        label="OpenAI 官方",
        class_name="ChatOpenAI",
        package="langchain-openai",
        env_vars=("OPENAI_API_KEY",),
        snippet='ChatOpenAI(model="gpt-4o-mini", temperature=0)',
    ),
    ModelProvider(
        key="openai_compatible",
        label="DeepSeek / Kimi / 通义（OpenAI 兼容协议）",
        class_name="ChatOpenAI",
        package="langchain-openai",
        env_vars=("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
        snippet=(
            "ChatOpenAI(\n"
            '    model="deepseek-chat",\n'
            '    base_url="https://api.deepseek.com/v1",\n'
            '    api_key=os.getenv("DEEPSEEK_API_KEY"),\n'
            ")"
        ),
        note="兼容协议的厂商都走这一类：只换 base_url 和 model，不用装新包",
    ),
    ModelProvider(
        key="azure_openai",
        label="Azure OpenAI",
        class_name="AzureChatOpenAI",
        package="langchain-openai",
        env_vars=("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
        snippet=(
            "AzureChatOpenAI(\n"
            '    azure_deployment="my-gpt-deployment",\n'
            '    api_version="2024-10-21",\n'
            ")"
        ),
        note="包和 ChatOpenAI 是同一个，只是类不同；api_version 必填",
    ),
    ModelProvider(
        key="anthropic",
        label="Anthropic Claude",
        class_name="ChatAnthropic",
        package="langchain-anthropic",
        env_vars=("ANTHROPIC_API_KEY",),
        snippet='ChatAnthropic(model="claude-sonnet-4-6", temperature=0)',
    ),
    ModelProvider(
        key="google_genai",
        label="Google Gemini",
        class_name="ChatGoogleGenerativeAI",
        package="langchain-google-genai",
        env_vars=("GOOGLE_API_KEY",),
        snippet='ChatGoogleGenerativeAI(model="gemini-2.5-flash")',
    ),
    ModelProvider(
        key="bedrock_converse",
        label="AWS Bedrock",
        class_name="ChatBedrockConverse",
        package="langchain-aws",
        env_vars=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"),
        snippet='ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6")',
    ),
    ModelProvider(
        key="ollama",
        label="本地 Ollama",
        class_name="ChatOllama",
        package="langchain-ollama",
        env_vars=(),
        snippet='ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")',
        note="本地模型不需要 API Key，但要先 ollama pull 把模型拉下来",
    ),
)


def list_model_inits() -> None:
    """列出 LangChain 调用各家模型的初始化方式。

    中文注释：这里只做静态对照，不真的去构造模型——所以哪怕某个包没装、
    某个 Key 没配，也能完整看到全部接入方式；每行末尾顺带标出当前可用状态，
    省得逐个翻文档确认。想真正拿到模型实例，把 snippet 抄进代码即可。
    """

    load_dotenv()

    print("LangChain 模型初始化方式清单")
    print("=" * 72)

    for index, provider in enumerate(_PROVIDERS, start=1):
        # 包名转模块名：langchain-openai -> langchain_openai
        module_name = provider.package.replace("-", "_")
        installed = "已安装" if find_spec(module_name) else "未安装"

        if not provider.env_vars:
            # Ollama 这类本地服务不需要 Key，单独标注，避免被误报成「没配置」
            env_state = "无需"
        else:
            missing = [name for name in provider.env_vars if not os.getenv(name)]
            env_state = "已配置" if not missing else "缺少 " + " / ".join(missing)

        # 多行代码片段统一缩进到冒号后面，否则打印出来会参差不齐
        snippet = provider.snippet.replace("\n", "\n" + " " * 14)

        print(f"\n{index}. {provider.label}")
        print(f"   provider : {provider.key}")
        print(f"   初始化类 : {provider.class_name}")
        print(f"   依赖包   : {provider.package}（{installed}）")
        print(f"   环境变量 : {', '.join(provider.env_vars) or '无需'}（{env_state}）")
        print(f"   初始化   : {snippet}")
        if provider.note:
            print(f"   注意     : {provider.note}")

    print("\n" + "=" * 72)
    print("统一入口：init_chat_model 按 provider 自动选类，但底层包仍需先安装")
    print('   init_chat_model("anthropic:claude-sonnet-4-6", temperature=0)')
    print('   init_chat_model("deepseek:deepseek-chat", temperature=0)')
    print("   未写 provider 时按模型名前缀推断，如 claude... -> anthropic")


if __name__ == "__main__":
    # 简单的手动分发：--models 看接入清单，--messages 跑消息用法演示，
    # 不带参数时仍然走 hello world
    if "--models" in sys.argv:
        list_model_inits()
    elif "--messages" in sys.argv:
        demo_messages()
    else:
        main()