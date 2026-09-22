# 作用：用 Chroma 查 top5，接口保持和 D21 的 search() 一样（返回 [(块, 相似度), ...]），这样上层的 llm/generator.py 和 main.py 一行都不用改 —— 这就是「换实现不换接口」。

import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from kb.embedder import embed_one
from kb.vector_store import get_collection

def search_chroma(question, top_k=5):
    q = embed_one(question)
    q = q / (np.linalg.norm(q) + 1e-9)          # 问题向量也归一化
    col = get_collection()
    res = col.query(
        query_embeddings=[q.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []  # 返回的是标签和分数
    # 返回的每个字段都是「一层列表套一层结果」：第 0 个内层才是我们这一次查询的
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(({**meta, "text": text}, 1 - dist))   # 余弦空间：相似度 = 1 - 距离  # **meta 这里已经有5个标签，再加一个
    return hits

if __name__ == "__main__":
    for q in ["打车费能报销吗", "公司 VPN 怎么用", "在职证明怎么开"]:
        print("=" * 60)
        print("问：", q)
        for c, s in search_chroma(q):
            print(f"  {s:.3f} | 《{c['doc_name']}》{c['section']}")
























