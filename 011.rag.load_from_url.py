"""RAG 第一步：文档加载（Document Loading）—— 从任意网页 URL 提取正文。

【这一步在干什么】
把网页上的原始内容转成 LangChain 认识的 Document 对象。
后面的切分、向量化、检索，全都建立在这个对象上。

【Document 就两样东西】
    page_content   正文文本
    metadata       元信息（来源 URL、标题、日期、作者……）
metadata 不是可有可无：检索回一段内容后，你要能回答「这句话出自哪个页面」。

【为什么不用 parse_only 指定 class】
指定 class/id 是把「某个页面此刻的 HTML 结构」写死，换个站点、甚至换个频道就抓不到。
实测新华网 tech 频道是 <div class="main-left left">，fortune 频道是 <div class="main-left">，
而且 SoupStrainer(class_=...) 是精确匹配整个 class 属性字符串 —— 少一个空格就 0 字。

正确做法：用 trafilatura 这类「正文提取」库自动判断哪块是正文。
它综合文本密度、标签语义、链接占比等特征打分，不需要人工指定任何选择器。

【加载 ≠ 切分】
加载器只负责「读出来」。一篇长文加载完还是 1 个大 Document，
把它切成小块是下一步的事。

跑法：python 011.rag.load_from_url.py
"""
import os

# 中文注释：副作用导入 —— 这个模块在被导入时设置 UA 环境变量、屏蔽 langchain-community
# 的 sunset 警告。langchain_community 是「导入时」就去读这些的，所以必须排在它前面。
# noqa: F401 是必要的：这个 import 确实没被使用，它的作用在执行导入本身。
import common.setup  # noqa: F401
import bs4
import requests
import trafilatura
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

# 中文注释：换成任何你想抓的网页都可以，不限于新闻站。
URLS = [
    "https://www.news.cn/tech/20260924/8472c1af7bbd40fbb953374091e08b0f/c.html",
    "https://www.news.cn/fortune/20261006/19af445d1dc84e84bac404c1304470fc/c.html",
    "https://www.python.org/about/",
    "https://www.ruanyifeng.com/blog/2024/01/weekly-issue-288.html",
]


def load_from_url(url: str) -> Document:
    """从任意网页提取正文，返回 Document。

    中文注释：核心是 trafilatura 自动识别正文，不依赖任何站点特定的 class/id。
    """
    response = requests.get(
        url, headers={"User-Agent": os.environ["USER_AGENT"]}, timeout=30
    )
    response.raise_for_status()

    # 中文注释：这里传的是 response.content（bytes），不是 response.text！
    # response.text 是 requests 按 HTTP 规范猜的编码解出来的：响应头若没带 charset，
    # 它会默认按 ISO-8859-1 解，中文页面直接乱码 —— 实测阮一峰的博客就是这样
    # （requests 猜 ISO-8859-1，实际是 utf-8，结果全篇变成「è¿éè®°å½」）。
    # 把 bytes 交给 trafilatura，它内部会自己探测编码，躲开这个坑。
    html_bytes = response.content

    # 中文注释：整页文本，只用来做对比 —— 让你直观看到正文提取砍掉了多少噪声。
    whole_text = bs4.BeautifulSoup(html_bytes, "lxml").get_text(strip=True)

    # 中文注释：两个参数值得注意
    #   include_comments 默认 True，会把读者评论一起抓进来，通常不是我们要的正文；
    #   include_tables 默认 True，表格里的数据往往就是正文，保留。
    body = trafilatura.extract(
        html_bytes,
        include_comments=False,
        include_tables=True,
    )
    if not body:
        # 中文注释：返回 None 说明它判断「这页没有正文」，常见于 JS 动态渲染的页面 ——
        # requests 只拿到空壳 HTML，正文是浏览器执行 JS 后才填进去的。
        raise ValueError(f"没能从 {url} 提取到正文（可能是 JS 动态渲染的页面）")

    # 中文注释：trafilatura 还能顺便解析出 title / date / author，省得自己写正则。
    meta = trafilatura.extract_metadata(html_bytes)

    return Document(
        page_content=body,
        metadata={
            "source": url,
            "title": meta.title if meta else None,
            "date": meta.date if meta else None,
            "author": meta.author if meta else None,
            "whole_page_length": len(whole_text),
        },
    )


def demo_compare(url: str) -> None:
    """同一个 URL 分别用「整页加载」和「正文提取」跑一遍，看差别。"""

    # 方式一：WebBaseLoader 原样加载整页，不加任何过滤
    whole = WebBaseLoader(web_path=[url]).load()[0].page_content

    # 方式二：trafilatura 自动提取正文
    doc = load_from_url(url)

    ratio = len(doc.page_content) / max(len(whole), 1)
    print(f"\n{'=' * 90}")
    print(f"URL      : {url}")
    print(f"整页加载 : {len(whole):>6} 字  开头 = {whole[:50]!r}")
    print(f"正文提取 : {len(doc.page_content):>6} 字  压缩到 {ratio:.0%}")
    print(f"metadata : title={doc.metadata['title']!r}")
    print(f"           date={doc.metadata['date']!r}  author={doc.metadata['author']!r}")
    print(f"正文开头 : {doc.page_content[:80]}")


if __name__ == "__main__":
    for target_url in URLS:
        try:
            demo_compare(target_url)
        except Exception as error:
            print(f"\n{'=' * 90}")
            print(f"URL      : {target_url}")
            print(f"失败     -> {type(error).__name__}: {error}")
