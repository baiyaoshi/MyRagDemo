"""
多模态图文检索演示
- 使用阿里云 Qwen3-VL-Embedding API 进行图文向量化
- 使用 Milvus 进行向量存储和检索
"""

import os
import base64
from tqdm import tqdm
from glob import glob
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image
import dashscope
from dashscope.embeddings.multimodal_embedding import (
    MultiModalEmbedding,
    MultiModalEmbeddingItemImage,
    MultiModalEmbeddingItemText,
)
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# ============================================================
# 1. 初始化设置
# ============================================================
# 阿里云 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

EMBEDDING_MODEL = "qwen3-vl-embedding"
EMBEDDING_DIM = 2560  # qwen3-vl-embedding 默认维度，可调范围 256~2560

# 获取当前脚本所在目录（绝对路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "visual_bge", "imgs")
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"


# ============================================================
# 2. 工具函数
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """将图片转为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_embedding(image_path: str) -> list[float]:
    """
    使用 Qwen3-VL-Embedding API 获取图片的向量表示。
    使用 MultiModalEmbeddingItemImage。
    """
    image_b64 = encode_image_to_base64(image_path)
    resp = MultiModalEmbedding.call(
        model=EMBEDDING_MODEL,
        input=[MultiModalEmbeddingItemImage(
            image=f"data:image/png;base64,{image_b64}",
            factor=1.0
        )],
        dimensions=EMBEDDING_DIM,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API 请求失败: {resp}")
    return resp.output["embeddings"][0]["embedding"]


def get_text_embedding(text: str) -> list[float]:
    """
    使用 Qwen3-VL-Embedding API 获取文本的向量表示（文本搜图片场景）。
    使用 MultiModalEmbeddingItemText。
    """
    resp = MultiModalEmbedding.call(
        model=EMBEDDING_MODEL,
        input=[MultiModalEmbeddingItemText(text=text, factor=1.0)],
        dimensions=EMBEDDING_DIM,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API 请求失败: {resp}")
    return resp.output["embeddings"][0]["embedding"]


def visualize_results(
    query_image_path: str,
    retrieved_images: list,
    img_height: int = 300,
    img_width: int = 300,
    row_count: int = 3
) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved

        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return np.hstack([query_display_area, panoramic_image])


# ============================================================
# 3. 初始化 Milvus 客户端
# ============================================================
print("--> 正在初始化 Milvus 客户端...")
milvus_client = MilvusClient(uri=MILVUS_URI)

# ============================================================
# 4. 创建 Milvus Collection
# ============================================================
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

# 获取图片列表
image_list = glob(os.path.join(DATA_DIR, "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR} 中未找到任何 .png 图像。")
print(f"找到 {len(image_list)} 张图片")

# 计算向量维度（先编码一张图获取维度）
dim = len(get_image_embedding(image_list[0]))
print(f"向量维度: {dim}")

# 定义字段和 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

schema = CollectionSchema(fields, description="多模态图文检索（Qwen3-VL-Embedding）")
print("Schema 结构:")
print(schema)

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

# ============================================================
# 5. 插入向量数据
# ============================================================
print(f"\n--> 正在编码图片并插入向量...")
data = []
for img_path in tqdm(image_list, desc="编码图片"):
    emb = get_image_embedding(img_path)
    data.append({
        "vector": emb,
        "image_path": img_path,
    })

milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
print(f"成功插入 {len(data)} 条数据")

# 创建索引
print("\n--> 正在创建 IVF_FLAT 索引...")
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="IP",  # 内积距离
    params={"nlist": 128}
)
milvus_client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params
)
print("索引创建完成")

# 加载 collection 到内存以便搜索
print("\n--> 正在加载 Collection 到内存...")
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("Collection 加载完成")

# ============================================================
# 6. 搜索与可视化
# ============================================================
print("\n--> 开始检索演示...")

# --- 6a. 图片搜图片 ---
print("【图片搜图片演示】")
query_image = image_list[0]  # 使用第一张图作为查询
query_vector = get_image_embedding(query_image)

search_params = {
    "metric_type": "IP",
    "params": {"nprobe": 10},
}
results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=6,
    search_params=search_params,
    output_fields=["image_path"],
)

print(f"\n查询图片: {query_image}")
print("检索结果:")
retrieved_images = []
for i, result in enumerate(results[0]):
    path = result["entity"]["image_path"]
    score = result["distance"]
    retrieved_images.append(path)
    print(f"  [{i}] {path} (相似度: {score:.4f})")

# 可视化结果
print("\n--> 正在生成可视化结果...")
panoramic = visualize_results(query_image, retrieved_images)
output_path = os.path.join(DATA_DIR, "retrieval_result_qwen3_vl.jpg")
cv2.imwrite(output_path, panoramic)
print(f"可视化结果已保存到: {output_path}")

# --- 6b. 文本搜图片 ---
print("\n【文本搜图片演示】")
text_query = "a dragon flying in the sky"
text_vector = get_text_embedding(text_query)

text_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[text_vector],
    limit=6,
    search_params=search_params,
    output_fields=["image_path"],
)

print(f"文本查询: '{text_query}'")
retrieved_by_text = []
for i, result in enumerate(text_results[0]):
    path = result["entity"]["image_path"]
    score = result["distance"]
    retrieved_by_text.append(path)
    print(f"  [{i}] {path} (相似度: {score:.4f})")

# 文本检索可视化
panoramic_text = visualize_results(query_image, retrieved_by_text)
output_text_path = os.path.join(DATA_DIR, "retrieval_result_text_query.jpg")
cv2.imwrite(output_text_path, panoramic_text)
print(f"文本检索可视化结果已保存到: {output_text_path}")

print("\n✅ 完成！")

