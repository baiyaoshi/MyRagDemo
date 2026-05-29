### 简单rag

```py
import os

from dotenv import load_dotenv
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

markdown_path="F:/develop/agent\MyRagDemo/markdown/cpu_high_debug.md"
loader = TextLoader(markdown_path,encoding="utf-8")
docs = loader.load()

#文本分块
text_splitter = RecursiveCharacterTextSplitter() #当不指定参数初始化 RecursiveCharacterTextSplitter() 时，其默认行为旨在最大程度保留文本的语义结构
"""默认分隔符与语义保留: 按顺序尝试使用一系列预设的分隔符 ["\n\n" (段落), "\n" (行), " " (空格), "" (字符)] 来递归分割文本。这种策略的目的是尽可能保持段落、句子和单词的完整性，因为它们通常是语义上最相关的文本单元，直到文本块达到目标大小。
保留分隔符: 默认情况下 (keep_separator=True)，分隔符本身会被保留在分割后的文本块中。
默认块大小与重叠: 使用其基类 TextSplitter 中定义的默认参数 chunk_size=4000（块大小）和 chunk_overlap=200（块重叠）。这些参数确保文本块符合预定的大小限制，并通过重叠来减少上下文信息的丢失。"""
texts = text_splitter.split_documents(docs)

#初始化中文嵌入模型（阿里云 DashScope）
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
#构建向量存储
vectorstore=InMemoryVectorStore(embeddings)
vectorstore.add_documents(texts)

#用户查询
question="文中有什么例子"
#找最相关三个
retrieved_docs=vectorstore.similarity_search(question,k=3)
#准备上下文
docs_content="\n\n".join(doc.page_content for doc in retrieved_docs)

prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

print(docs_content)

answer = llm.invoke(prompt.format(context=docs_content,question=question))
print(answer.content)
```



### unstructured数据加载

```py
from unstructured.partition.auto import partition

pdf_path="F:/develop/agent/MyRagDemo/pdf/2024软研笔试题.pdf"
elements=partition(
    filename=pdf_path,
    content_type="application/pdf"
)
print(f"解析完成{len(elements)}个元素，{sum(len(str(e))for e in elements) }个字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
```

**partition 函数参数解析：**

- `filename`: 文档文件路径，支持本地文件路径；
- `content_type`: 可选参数，指定MIME类型（如"application/pdf"），可绕过自动文件类型检测；
- `file`: 可选参数，文件对象，与 filename 二选一使用；
- `url`: 可选参数，远程文档 URL，支持直接处理网络文档；
- `include_page_breaks`: 布尔值，是否在输出中包含页面分隔符；
- `strategy`: 处理策略，可选 "auto"、"fast"、"hi_res" 等；
- `encoding`: 文本编码格式，默认自动检测。

`partition`函数使用自动文件类型检测，内部会根据文件类型路由到对应的专用函数（如PDF文件会调用`partition_pdf`）。如果需要更专业的PDF处理，可以直接使用`from unstructured.partition.pdf import partition_pdf`，它提供更多PDF特有的参数选项，如OCR语言设置、图像提取、表格结构推理等高级功能，同时性能更优。



## 文本分块

### 递归字符分块

```
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 分隔符优先级
    chunk_size=200,
    chunk_overlap=10,
)
```

### 语义切割

langchain_experimental.text_splitter.SemanticChunker

```py
embeddings=DashScopeEmbeddings(
    model="text-embedding-v2"
)
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # 断点识别方法
)
loader = TextLoader("../../txt/蜂医.txt")
documents=loader.load()

docs = text_splitter.split_documents(documents)
```

**断点识别方法 (`breakpoint_threshold_type`)**

如何定义“显著的语义跳跃”是语义分块的关键。`SemanticChunker` 提供了几种基于统计的方法来识别断点：

- `percentile` (百分位法 - **默认方法**):
  - **逻辑**: 计算所有相邻句子的语义差异值，并将这些差异值进行排序。当一个差异值超过某个百分位阈值时，就认为该差异值是一个断点。
  - **参数**: `breakpoint_threshold_amount` (默认为 `95`)，表示使用第95个百分位作为阈值。这意味着，只有最显著的5%的语义差异点会被选为切分点。
- `standard_deviation` (标准差法):
  - **逻辑**: 计算所有差异值的平均值和标准差。当一个差异值超过“平均值 + N * 标准差”时，被视为异常高的跳跃，即断点。
  - **参数**: `breakpoint_threshold_amount` (默认为 `3`)，表示使用3倍标准差作为阈值。
- `interquartile` (四分位距法):
  - **逻辑**: 使用统计学中的四分位距（IQR）来识别异常值。当一个差异值超过 `Q3 + N * IQR` 时，被视为断点。
  - **参数**: `breakpoint_threshold_amount` (默认为 `1.5`)，表示使用1.5倍的IQR。
- `gradient` (梯度法):
  - **逻辑**: 这是一种更复杂的方法。它首先计算差异值的变化率（梯度），然后对梯度应用百分位法。对于那些句子间语义联系紧密、差异值普遍较低的文本（如法律、医疗文档）特别有效，因为这种方法能更好地捕捉到语义变化的“拐点”。
  - **参数**: `breakpoint_threshold_amount` (默认为 `95`)。