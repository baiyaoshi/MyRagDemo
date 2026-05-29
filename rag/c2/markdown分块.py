"""
演示 MarkdownHeaderTextSplitter 的两种用法：
1. 基本用法：仅按标题结构分块
2. 组合用法：按标题分块后，再结合 RecursiveCharacterTextSplitter 进行细分
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ============================================================
# 准备一个示例 Markdown 文档
# ============================================================
markdown_document = """
# 第一章：机器学习基础

## 1.1 什么是机器学习

机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进，而无需进行明确的编程。

### 监督学习

监督学习使用标记的训练数据来学习从输入到输出的映射关系。常见的算法包括线性回归、决策树、支持向量机等。

### 无监督学习

无监督学习处理没有标记的数据，目标是发现数据中隐藏的结构或模式。聚类、降维是典型的无监督学习任务。

## 1.2 模型评估

模型评估是机器学习流程中至关重要的步骤，它帮助我们了解模型的性能和泛化能力。

### 评估指标

对于分类问题，常用的评估指标包括准确率、精确率、召回率和 F1 分数。对于回归问题，则常用均方误差（MSE）和平均绝对误差（MAE）。

### 交叉验证

交叉验证是一种评估模型泛化性能的统计方法，它将数据集分成多个子集，在多个子集组合上进行训练和验证。

# 第二章：深度学习入门

## 2.1 神经网络基础

神经网络由大量的神经元（节点）相互连接而成，每个连接都有对应的权重。

### 激活函数

常见的激活函数包括 Sigmoid、Tanh、ReLU 等。ReLU 因其计算简单且能缓解梯度消失问题而被广泛使用。

### 前向传播与反向传播

前向传播计算网络的输出，反向传播则根据损失函数计算梯度并更新网络权重。

## 2.2 卷积神经网络

卷积神经网络（CNN）专门用于处理具有网格状拓扑结构的数据，如图像。

### 卷积层

卷积层使用卷积核在输入数据上滑动，提取局部特征。

### 池化层

池化层用于降低特征图的维度，减少计算量，同时保持重要特征。
"""

# ============================================================
# 用法一：仅按标题结构分块
# ============================================================
print("=" * 60)
print("【用法一：仅按标题结构分块】")
print("=" * 60)

# 定义标题层级映射规则
headers_to_split_on = [
    ("#", "一级标题"),   # H1 -> 元数据 key 为 "一级标题"
    ("##", "二级标题"),  # H2 -> 元数据 key 为 "二级标题"
    ("###", "三级标题"), # H3 -> 元数据 key 为 "三级标题"
]

# 创建 MarkdownHeaderTextSplitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# 执行分块
md_header_splits = markdown_splitter.split_text(markdown_document)

print(f"共分割出 {len(md_header_splits)} 个文本块\n")

for i, chunk in enumerate(md_header_splits, 1):
    print(f"--- 块 {i} ---")
    print(f"元数据: {chunk.metadata}")
    print(f"内容预览（前 100 字）: {chunk.page_content[:100]}...")
    print()

# ============================================================
# 用法二：与 RecursiveCharacterTextSplitter 组合使用
# ============================================================
print("=" * 60)
print("【用法二：组合 RecursiveCharacterTextSplitter 细分】")
print("=" * 60)

# 第一步：先用 MarkdownHeaderTextSplitter 按标题分块（这次只按一级和二级标题分）
headers_to_split_on_v2 = [
    ("#", "一级标题"),
    ("##", "二级标题"),
]

markdown_splitter_v2 = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on_v2
)

# 获取较大的逻辑块
logical_chunks = markdown_splitter_v2.split_text(markdown_document)
print(f"第一步 - 按标题分块得到 {len(logical_chunks)} 个逻辑块\n")
print("++++++++++++++++++++")
print(logical_chunks)
# 第二步：对每个逻辑块再用 RecursiveCharacterTextSplitter 细分
# 注意：这些细分后的块会自动继承第一步的标题元数据
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,          # 每个块的最大字符数
    chunk_overlap=30,        # 块之间的重叠字符数
    length_function=len,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)

final_chunks = char_splitter.split_documents(logical_chunks)
print(f"第二步 - 进一步细分后得到 {len(final_chunks)} 个最终块\n")

for i, chunk in enumerate(final_chunks, 1):
    print(f"--- 最终块 {i} ---")
    print(f"元数据: {chunk.metadata}")
    print(f"内容: {chunk.page_content}")
    print()
