"""标准内容块（Content Blocks）——一次写代码，适配各家模型。

【解决什么问题】
每家供应商返回的 content 格式都不一样：

- OpenAI 风格：content 是字符串，或 [{"type": "text", ...}, {"type": "image_url", ...}]
- Anthropic 风格：content 是 [{"type": "text", ...}, {"type": "image", "source": {...}}]

结果就是：换个模型，光把图片塞进消息这段代码就得重写一遍。

【Content Blocks 是什么】
LangChain 1.0 给每条消息加了 .content_blocks 属性，把各家格式归一化成同一套结构：

    message.content          # 原始格式，随供应商而变（可能是 str，也可能 list[dict]）
    message.content_blocks   # 标准化格式，跨供应商一致

拿的时候用它，就不用关心对面是谁；写的时候也能直接用它构造消息，
LangChain 会在真正发请求时翻译回该供应商认识的格式。

跑法：python content_blocks.py
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.content import create_image_block, create_text_block

from common.utils import build_model


# ---------- 1. 块类型一览 ----------
# 这些都是 langchain-core >= 1.0 认定的标准块类型。
# 如果某个块的 type 不在这张表里，它就被当成供应商私有格式，会落到 non_standard。
STANDARD_BLOCKS: tuple[tuple[str, str], ...] = (
    ("text", "文本输出，最常用的一个"),
    ("reasoning", "推理过程，只有推理模型才给（如 deepseek-reasoner）"),
    ("image", "图片：url / base64 + mime_type / file_id 三种来源"),
    ("audio", "音频"),
    ("video", "视频"),
    ("file", "文件，如 PDF"),
    ("text-plain", "纯文本文件的内容"),
    ("tool_call", "模型申请调用工具，只会出现在 AIMessage 上"),
    ("tool_call_chunk", "流式输出时工具调用的分片"),
    ("server_tool_call", "供应商在服务端执行的工具调用"),
    ("server_tool_result", "上面那个服务端工具的执行结果"),
    ("non_standard", "供应商私有格式的兜底，LangChain 不认识就归到这儿"),
)


# ---------- 2. 自己造一条多模态消息（纯本地，不联网） ----------
# 中文注释：这里用三种写法构造同一件事，实际项目挑顺手的就行。
# 关键是它们最终都归一化成同一份 content_blocks。
def build_multimodal_message() -> HumanMessage:
    """构造一条「文字 + 图片」的多模态消息。"""

    # 写法 A：直接写 dict，跟标准块结构一一对应，最直观
    by_dict = HumanMessage(
        content_blocks=[
            {"type": "text", "text": "这张图里是什么？"},
            {"type": "image", "url": "https://example.com/cat.png"},
        ]
    )

    # 写法 B：用工厂函数，会自动生成 id，且必填参数在构造时就校验
    by_factory = HumanMessage(
        content_blocks=[
            create_text_block("这张图里是什么？"),
            create_image_block(url="https://example.com/cat.png"),
        ]
    )

    # 写法 C：图片也可以用 base64 传（本地文件、截图常用这种）。
    # 注意 base64 必须同时给 mime_type，否则供应商不知道该按什么格式解码。
    by_base64 = HumanMessage(
        content_blocks=[
            {"type": "text", "text": "识别这张图里的文字。"},
            {"type": "image", "base64": "<base64 字符串>", "mime_type": "image/png"},
        ]
    )

    print("   A 写法 dict     :", by_dict.content_blocks)
    print("   B 写法 工厂函数 :", by_factory.content_blocks)
    print("   C 写法 base64   :", by_base64.content_blocks)
    print("   三种写法构造出的 content 与 content_blocks 一致：",
          by_dict.content == by_dict.content_blocks)

    return by_dict


# ---------- 3. 读取：content_blocks 会把字符串也归一化 ----------
# 中文注释：这条是理解 content_blocks 的关键——哪怕内容只是个普通字符串，
# content_blocks 也会把它包成 [{"type": "text", "text": ...}]。
# 所以读的时候可以放心按「块的列表」处理，不用先判断 content 是 str 还是 list。
def show_normalization() -> None:
    """演示字符串内容被归一化成文本块。"""

    plain = HumanMessage("你好")
    print("   .content        :", repr(plain.content))
    print("   .content_blocks :", plain.content_blocks)

    # 带推理过程的回复长这样（这里手工造一份，方便观察结构）
    reasoning = AIMessage(
        content_blocks=[
            {"type": "reasoning", "reasoning": "先算 1+1，再检查一遍。"},
            {"type": "text", "text": "答案是 2。"},
        ]
    )
    print("   推理模型的回复  :", reasoning.content_blocks)
    print("   注意 content 也被同步填成了同样内容：", reasoning.content)


# ---------- 4. 实战：按块类型分发处理 ----------
def extract_text(message) -> str:
    """从任意消息里抽出纯文本。

    中文注释：这就是 content_blocks 最实用的地方——不管对面是哪家模型、
    返回的是字符串还是复杂的内容块，这里只挑 type == "text" 的块拼起来，
    换模型一行都不用改；推理块、图片块被自然忽略。

    Args:
        message: 任意 BaseMessage（HumanMessage / AIMessage / AIMessageChunk 都行）。

    Returns:
        所有文本块拼接后的字符串。
    """

    return "".join(
        block.get("text", "")
        for block in message.content_blocks
        if block["type"] == "text"
    )


def count_by_type(message) -> dict[str, int]:
    """统计一条消息里各类型块的数量，调试多模态输出时很有用。"""

    counts: dict[str, int] = {}
    for block in message.content_blocks:
        counts[block["type"]] = counts.get(block["type"], 0) + 1
    return counts


# ---------- 5. 真实调用：看模型实际返回什么 ----------
def call_model() -> None:
    """真实请求一次，观察返回消息的 content_blocks。"""

    model = build_model()

    reply = model.invoke([HumanMessage("用一句话说明什么是 token")])
    print("   .content        :", repr(reply.content)[:80], "...")
    print("   .content_blocks :", reply.content_blocks)
    print("   块类型统计      :", count_by_type(reply))
    print("   extract_text()  :", extract_text(reply)[:60], "...")

    # 流式场景同理：chunk 用 + 拼成完整消息后，content_blocks 也一样可用
    merged = None
    for chunk in model.stream([HumanMessage("数到三")]):
        merged = chunk if merged is None else merged + chunk
    print("   流式合并后的块  :", merged.content_blocks)


if __name__ == "__main__":
    print("1. 标准块类型：" + "、".join(name for name, _ in STANDARD_BLOCKS))
    print()

    print("2. 自己造一条多模态消息：")
    build_multimodal_message()
    print()

    print("3. content_blocks 会把普通字符串也归一化：")
    show_normalization()
    print()

    print("4. 真实模型返回：")
    call_model()
    print()

    print("5. extract_text 对多模态消息同样有效（自动忽略图片块）：")
    multimodal = HumanMessage(
        content_blocks=[
            create_text_block("这张图里是什么？"),
            create_image_block(url="https://example.com/cat.png"),
        ]
    )
    print("  ", extract_text(multimodal))
