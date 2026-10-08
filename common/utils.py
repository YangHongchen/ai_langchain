"""通用工具方法。

中文注释：各个 demo 文件都以数字开头（001.main.py、002.message.py ...），
这种文件名不是合法的 Python 模块名，没法用 import 相互引用，
所以把共用的模型构造提到这里，由 common 包统一对外提供。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

# 中文注释：DeepSeek 只提供对话接口，没有 embeddings 接口（实测 /embeddings 返回 404），
# 所以向量化改用本地 ONNX 模型，权重放在项目根的 models/ 下，跑起来不需要联网。
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bge-small-zh-v1.5"


def build_model() -> ChatOpenAI:
    """构造演示要用的模型实例。

    中文注释：各个 demo 共用同一份模型配置，抽出来就只需改这一处。
    注意 Key 走的是自定义的 DEEPSEEK_API_KEY，而 ChatOpenAI 默认只认环境变量
    OPENAI_API_KEY —— 换了变量名就必须显式传 api_key，
    否则运行时会报 "The api_key client option must be set"。
    """

    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填写 API Key。"
        )

    # DeepSeek 兼容 OpenAI 协议，因此直接复用 ChatOpenAI，
    # 只需把 base_url 指向 DeepSeek 网关。
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
        api_key=api_key,
        temperature=0,
        timeout=60,
    )


class LocalFastEmbed(Embeddings):
    """把 fastembed 的本地 ONNX 模型接到 LangChain 的 Embeddings 接口上。

    中文注释：LangChain 自带的 FastEmbedEmbeddings 只接受「模型名」，内部会去
    HuggingFace 拉权重；本机连不通 HF（TCP 超时），只能改用 fastembed 的
    specific_model_path 指向已经下好的目录 —— 而这个参数官方封装没有透传出来，
    所以这层薄适配只能自己写。

    接口就两个方法，语义要分清：建库/切分用 embed_documents，检索时用 embed_query。
    """

    def __init__(self, model_dir: Path) -> None:
        if not model_dir.is_dir():
            raise RuntimeError(
                f"本地 embedding 模型不存在：{model_dir}\n"
                "请先从 ModelScope 仓库 Qdrant/bge-small-zh-v1.5 下载 config.json、"
                "tokenizer.json、tokenizer_config.json、special_tokens_map.json、"
                "model_optimized.onnx 放到该目录。"
            )
        self._model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            specific_model_path=str(model_dir),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # 中文注释：query_embed 会给查询句补上 BGE 官方推荐的检索前缀（如「为这个句子
        # 生成表示以用于检索相关文章：」），和 embed 的产出不是一回事，两边别混用。
        return next(self._model.query_embed(text)).tolist()


def build_embeddings() -> Embeddings:
    """构造本地中文 embedding 模型（BAAI/bge-small-zh-v1.5，512 维，无需联网）。"""
    return LocalFastEmbed(EMBEDDING_MODEL_DIR)

