# 建向量库
import sys,os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)

import json
import numpy as np
import chromadb

DB_DIR = "chroma_db"    # 数据库罗盘的目录
COLLECTION = "kb_chunks"    # 集合名：只能字母数字._-,不能中文

def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR) # 持久化：数据真正写进磁盘
    return client.get_or_create_collection(  # 创建一张表
        name=COLLECTION,  # 表的名字
        embedding_function=None,  # 用我自己计算，不用向量默认
        configuration={"hnsw":{"space":"cosine"}} # 用近似搜索跳着，和余弦相似度 
    )

def build(chunk_path="kb/chunks.json", vec_path="kb/vectors.npy"):
    chunks = json.load(open(chunk_path, encoding="utf-8"))
    vectors = np.load(vec_path)
    # 入库前统一归一化：向量长度变成 1，余弦距离与点积口径统一（同 D21 的做法）
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

    col = get_collection()
    col.upsert(                                             # upsert = 有则覆盖、无则新增，脚本可重复跑
        ids=[f"chunk-{c['chunk_id']:04d}" for c in chunks], # 加工214个块 产出214个列表 ，把变量塞进字符串模版 格式说明符
        documents=[c["text"] for c in chunks],  # 
        embeddings=vectors.tolist(),  # 这里对应上文的 embedding_function=None 转成pyhon的嵌套列表
        metadatas=[{
            "chunk_id": c["chunk_id"],
            "doc_name": c["doc_name"],
            "section": c["section"],
            "category": c["category"],
            "source": c["source"],
        } for c in chunks],
    )
    print(f"入库 {col.count()} 块 → {DB_DIR}/（集合名 {COLLECTION}）")

if __name__ == "__main__":
    build()



























