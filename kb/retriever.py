# 手写检索
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import json
import numpy as np
from kb.embedder import embed_one

def load_kb(chunk_path="kb/chunks.json",vec_path="kb/vectors.npy"):
    chunks = json.load(open(chunk_path,encoding="utf-8"))
    vectors = np.load(vec_path)
    return chunks,vectors
    # 切片和向量

def search(question,chunks,vectors,top_k=5):
    q = embed_one(question) # 把问题也变成向量
    q = q / (np.linalg.norm(q) + 1e-9) # 归一化（长度变1：加极小值防除以0）
    # 分母 向量长度加上 10亿分之1 也就是归一化
    V = vectors / (np.linalg.norm(vectors,axis=1,keepdims=True)+ 1e-9)
    # 这里也是归一化 axis=1 每一行的长度1 不写就是整个矩阵 keepdims=True 
    scores = V @ q     # 归一化后，点积 = 余弦相似度保持矩阵的维度形状
    # 说人话就是问题只有一块 也就是（1024，） V不一样 他是214块 即（214,1024）
    idx = np.argsort(scores)[::-1][:top_k]  # 从大到小取前top_k个 按分数从小到大排序 要前五个
    return [(chunks[i],float(scores[i])) for i in idx] # 遍历并返回一对 [(块1, 0.83), (块2, 0.79), (块3, 0.71), ...]


