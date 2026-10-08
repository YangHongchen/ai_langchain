"""统一的环境准备：设置 UA、屏蔽第三方库的导入期警告。

【为什么需要这个模块】
langchain_community 在【导入时】就会做两件事：
  1. 读 USER_AGENT 环境变量（读不到就打警告，且之后再设无效）
  2. 打一条「已被官方标记 sunset」的 DeprecationWarning
也就是说这两项设置必须抢在 `import langchain_community` 之前生效。

langchain_experimental 也有同样的问题：它的 __init__.py 一被导入就 warn 一条 sunset
提示。而 SemanticChunker 目前【只】存在于这个包里（langchain_text_splitters 里没有），
躲不开，只能把噪音过滤掉。

但 PEP 8 要求所有 import 都放在文件头 —— 两件事直接冲突。
解法是把「设置动作」封装成本模块的导入副作用，然后在使用方把它排在第三方 import 之前：

    import common.setup          # noqa: F401  副作用导入，必须排在最前
    import bs4
    from langchain_community.document_loaders import TextLoader

Python 是按顺序执行 import 的，所以 common.setup 会先跑完，再把控制权交给后面的库。

【约束】
本模块自身不能导入任何 langchain 相关的包 —— 否则就绕回同一个问题了。
"""
import os
import warnings

# 中文注释：很多站点会拒绝空 UA 或 python-requests 的默认 UA，设一个有辨识度的。
USER_AGENT = "langchain-learning-demo/1.0"
os.environ.setdefault("USER_AGENT", USER_AGENT)

# 中文注释：这两条警告都只是「包的状态声明」（官方标记 sunset、不再积极维护），
# 不代表 API 不能用，屏蔽掉让演示输出干净。
# 将来真要迁移时，把对应那行删掉就能重新看到提示。
warnings.filterwarnings("ignore", message=r".*langchain-community.*is being sunset.*")
warnings.filterwarnings("ignore", message=r".*langchain-experimental.*is being sunset.*")
