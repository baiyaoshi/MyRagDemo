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