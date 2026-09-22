# kb/build_index.py
# 一键重建索引：docs/*.md → 切块 → 向量化 → 写入 Chroma
# 仓库不提交二进制索引，别人 clone 后跑这一条命令就能建起来



import sys,os,json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)

import numpy as np
from kb.loader import load_docs
from kb.splitter import split_docs
from kb.embedder import embed_texts
from kb.vector_store import build


CHUNK_PATH = "kb/chunks.json"
VEC_PATH = "kb/vectors.npy"

def main():
    # ① 读文档
    docs = load_docs()
    print(f"[1/4] 读到 {len(docs)} 份文档")

     # ② 切块，同时落盘 chunks.json
    chunks = split_docs(docs)
    with open(CHUNK_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[2/4] 切出 {len(chunks)} 块 → {CHUNK_PATH}")

    # ③ 向量化（这一步要花钱，调 DashScope text-embedding-v3）
    vectors = embed_texts([c["text"] for c in chunks])
    np.save(VEC_PATH, vectors)
    print(f"[3/4] 向量化完成 {vectors.shape} → {VEC_PATH}")

    # ④ 写进 Chroma
    build(chunk_path=CHUNK_PATH, vec_path=VEC_PATH)
    print("[4/4] 索引就绪，可以跑了")


if __name__ == "__main__":
    main()































