"""RAG
第一步：文档加载（Document Loading）—— 从 Word（.docx）读取文字。
"""
import zipfile
from pathlib import Path
from xml.etree import ElementTree

# 中文注释：副作用导入 —— 这个模块在被导入时设置 UA 环境变量、屏蔽 langchain-community
# 的 sunset 警告。langchain_community 是「导入时」就去读这些的，所以必须排在它前面。
# noqa: F401 是必要的：这个 import 确实没被使用，它的作用在执行导入本身。
import common.setup  # noqa: F401
from langchain_community.document_loaders import Docx2txtLoader

# 中文注释：改成你自己的 docx 路径即可。这个文件只读，不会被修改。
DOCX_FILE = Path("assets/sample.docx")


def load_from_word():
    """加载本地 .docx，返回 list[Document]。"""

    if not DOCX_FILE.exists():
        print(f"找不到 {DOCX_FILE.resolve()}")
        return []

    # 中文注释：docx 不需要 encoding —— 它本质是个 zip 包，正文在 word/document.xml 里
    # 是 XML，编码由文件自己声明。也不需要装 Office。
    #
    # 和 PDF 的区别：docx 没有「页」这个概念（分页是渲染时才计算出来的），
    # 所以整个文件只产出 1 个 Document，不像 PyPDFLoader 那样每页一个。
    docs = Docx2txtLoader(str(DOCX_FILE)).load()

    doc = docs[0]

    # 中文注释：Docx2txtLoader 只填了 source，作者、时间这些它不解析。
    # 但那些信息就在 docx 内部的 docProps/core.xml 里，自己读一下就有 ——
    # RAG 里「这段内容是谁、什么时候写的」往往是有用的检索维度。
    doc.metadata.update(read_docx_properties(DOCX_FILE))

    print(f"共 {len(docs)} 个 Document，{len(doc.page_content)} 字")
    print(f"  metadata :")
    for key, value in doc.metadata.items():
        print(f"    {key} = {value!r}")
    print(f"  正文全文 :")
    print(doc.page_content)

    return docs


def read_docx_properties(path: Path) -> dict:
    """从 docx 内部的 docProps/core.xml 读出作者、创建/修改时间等元数据。

    中文注释：用标准库就能做 —— docx 就是个 zip，不需要额外的解析依赖。
    """
    # 中文注释：打开 zip 只读，不改动文件。
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("docProps/core.xml")

    # 中文注释：用 ElementTree 而不是正则 —— XML 的命名空间和实体转义都交给它处理。
    root = ElementTree.fromstring(xml_bytes)

    # 中文注释：core.xml 用了三套命名空间，查元素时必须带上前缀对应的 URI，
    # 不能直接写 "dc:creator" 这样的字面量。
    namespaces = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
        "dcterms": "http://purl.org/dc/terms/",
    }
    return {
        "author": root.findtext("dc:creator", namespaces=namespaces),
        "last_modified_by": root.findtext("cp:lastModifiedBy", namespaces=namespaces),
        "created": root.findtext("dcterms:created", namespaces=namespaces),
        "modified": root.findtext("dcterms:modified", namespaces=namespaces),
    }


if __name__ == "__main__":
    load_from_word()
