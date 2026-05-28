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

answer = llm.invoke(prompt.format(context=docs_content,question=question))
print(answer.content)

