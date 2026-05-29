from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from unstructured import documents

from rag.c1.rag01 import embeddings

load_dotenv()

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

