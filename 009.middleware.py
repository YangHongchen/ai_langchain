"""中间件（Middleware）：在 agent 循环里插钩子。

不用改动 agent 或工具的代码，就能拦截 / 修改 / 增强请求和响应。

【关键】中间件不是回调函数，而是被编译进 LangGraph 图的真实节点 ——
触发时机和执行次数由「图的边」决定，不由你调用它们的顺序决定：
    before_model / after_model  挂在模型节点上，模型循环几轮就触发几次
                                （问一个问题，模型调了 2 次，钩子就跑了 2 次）
    wrap_tool_call              挂在工具调用上，一次申请几个工具就触发几次
    需要「整轮只跑一次」→ 用 before_agent / after_agent

所以别在 before_model 里做「一个提问只该做一次」的事（扣费、写审计日志）。

agent 跑一轮的位置图，方括号就是能插钩子的地方：
    用户输入 → [before_agent] → 调模型 → [after_model]
             → 要调工具？→ [wrap_tool_call] → 回到「调模型」
             → 不用调了 → [after_agent] → 输出

【两类钩子：改「状态」还是改「这一次请求」】
    (state, runtime)    before_agent / after_agent / before_model / after_model
                        拿到的是状态快照。改动会累积进对话历史，一直传下去；
                        只能看和改，不能阻止调用。
    (request, handler)  wrap_model_call / wrap_tool_call
                        拿到的是本次请求 + 执行权。改动出栈即消失，不进历史；
                        手里有 handler，所以能阻断（不调它）、能重试（调多次）、
                        能换模型/工具/提示（request.override）。

    判断标准：这个改动该不该留在对话历史里？
        该留（检索资料要塞进上下文）→ before_ / after_
        不该留（临时换模型、打点日志）→ wrap_
    坑：同一个东西别两层都改 —— wrap 层的 override 会盖掉 state 层的修改。

@before_model / @after_model / @wrap_tool_call 是装饰器写法，一个函数一个钩子；
需要自定义 state 或多个钩子组合时，再继承 AgentMiddleware。

跑法：python 009.middleware.py
"""
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from common.utils import build_model


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city}今天晴，26℃。"


@tool
def delete_user(user_id: str) -> str:
    """删除指定用户。"""
    return f"已删除用户 {user_id}。"


@before_model
def log_state(state, runtime):
    """调模型之前钩子：入参是 state 和 runtime。"""
    print(f"[before_model] 当前消息数 {len(state['messages'])}")


@wrap_tool_call
def guard(request, handler):
    """工具执行钩子：request 是本次调用，handler 是真正的执行。"""
    name = request.tool_call["name"]

    # 调不调 handler 由你决定 —— 这就是拦截能力的来源
    if name == "delete_user":
        print(f"[wrap_tool_call] 拦截 {name}")
        return ToolMessage(
            content="危险操作已被安全策略拦截。",
            tool_call_id=request.tool_call["id"],
        )

    print(f"[wrap_tool_call] 放行 {name}")
    return handler(request)


agent = create_agent(
    build_model(),
    tools=[get_weather, delete_user],
    middleware=[log_state, guard],
)

for question in ("北京天气怎么样？", "把用户 u123 删掉"):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"问：{question}")
    print(f"答：{result['messages'][-1].text}\n")
