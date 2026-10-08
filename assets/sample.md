# LangChain RAG 实践笔记

> 本文记录 RAG 链路的四个阶段与常见坑，用于文档切分与检索测试。
> 内容主题：文档加载 → 文档切分 → 向量化入库 → 检索生成。

RAG（Retrieval-Augmented Generation，检索增强生成）的核心思路不是「让模型学会知识」，
而是**在提问的那一刻，把相关的知识递到它眼前**。

## 一、文档加载

### 1.1 加载器的粒度差异

不同格式交给你的 `Document` 数量完全不同，这直接决定后面的切分策略：

| 格式 | 加载器 | 粒度 | 依赖包 |
| --- | --- | --- | --- |
| 纯文本 .txt | `TextLoader` | 一个文件一个 Document | 无 |
| PDF | `PyPDFLoader` | **每页一个** | pypdf |
| Word .docx | `Docx2txtLoader` | 整个文件一个 | docx2txt |
| Excel .xlsx | `openpyxl` 自行读取 | **一行一个** | openpyxl |
| 网页 | `trafilatura` 提取正文 | 一页一个 | trafilatura |

其中 Excel 最特殊：它不是「一篇文章」，而是「多条独立记录」。
如果把整张表塞成一个 Document，检索时命中了也不知道是哪一行。

### 1.2 编码问题

txt 和网页都会遇到编码坑，而且症状都是**乱码**：

- txt：`encoding` 参数不写就用系统默认编码。普通中文 Windows 默认 `cp936`，
  读 UTF-8 的中文文件会乱码甚至抛 `RuntimeError`。
- 网页：响应头没带 `charset` 时，`requests` 按 HTTP 规范默认猜 `ISO-8859-1`，
  中文页面直接变乱码。解法是把 `response.content`（bytes）交给下游自己探测编码。

> 通用原则：**不确定编码时，不要把 bytes 自己解码，交给能探测编码的库处理。**

### 1.3 metadata 不是可有可无

检索回一段内容后，你必须能回答「这句话出自哪个文件、哪一页、哪一行」。
所以加载阶段就要把来源信息写进 `metadata`，切片时它会自动继承。

## 二、文档切分

### 2.1 为什么要切

不切有三个问题：

1. **塞不进** —— 上下文窗口有上限，长文档放不下
2. **不精准** —— 用户问一个细节，召回的是整篇，向量被平均化后谁都不像
3. **太费钱** —— 输入按 token 计费，每次都塞整篇等于重复付费

### 2.2 三个参数

把一篇长文抄到多张纸上，抄满一张就换下一张：

- `chunk_size`：一张纸能抄多少字
- `separators`：在哪下刀，聪明的话停在句号逗号处
- `chunk_overlap`：换纸时往回多抄几个字，防止切口把一句话劈成两半

### 2.3 中文必须自定义 separators

`RecursiveCharacterTextSplitter` 的默认分隔符是 `["\n\n", "\n", " ", ""]`，
这是**为英文设计的**——英文靠空格分词。中文不用空格，会一路降级到逐字符硬切。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
```

### 2.4 取值建议

| 参数 | 建议值 | 说明 |
| --- | --- | --- |
| `chunk_size` | 中文 300~800 字 | 一片约等于一到两个自然段 |
| `chunk_overlap` | 10%~20% | 不超过 200；设成和 chunk_size 一样大最糟 |
| `separators` | 中文标点 | 必须自定义 |

注意 `chunk_size` 数的是**字符数**，不是 token 数。

## 三、向量化入库

把每个 chunk 送进 embedding 模型，得到向量后存进向量库。这一步的关键是：

- 同一个库里的所有向量**必须来自同一个 embedding 模型**
- 换模型等于重建整个库
- 入库时 metadata 一并存好，检索后要能取回来

## 四、检索生成

检索到相关片段后，把它们拼进提示词，交给模型生成回答。

```
用户提问 → 向量检索召回 Top-K 片段 → 拼进提示词 → 模型生成
```

## 五、常见坑清单

1. **用英文默认 separators 切中文** —— 句子被劈开，检索匹配不上
2. **`chunk_size` 设得太小** —— 以为越小越准，实际把语义拆散了
3. **`chunk_overlap` 设得过大** —— 片数暴增，存了一堆近似重复的向量
4. **忽略 metadata** —— 召回了内容却说不出来源
5. **编码不显式指定** —— 换台机器就乱码
