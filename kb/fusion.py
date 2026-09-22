# RRF 融合：把两路检索的「排名」合成一路
# 建FRE融合器
import sys,os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)

from kb.retriever_chroma import search_chroma
from kb.retriever_bm25 import search_bm25

K = 60          # 平滑常数：k 越大，名次差异被压得越平；60 是原论文推荐值
POOL = 20       # 每路先各取 20 条进「候选池」，融合后再取前 top_k

def rrf_fuse(rank_lists, k=K):
    """rank_lists: [[(块, 分数), ...], ...] —— 传入多路排名，返回融合后的 [(块, rrf分数), ...]"""
    scores = {}
    pool ={}
    for hits in rank_lists:
        for rank, (c,_s) in enumerate(hits,start=1): # 名词从一开始，不是从0
            
            cid = c["chunk_id"]
            pool[cid] = c
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(pool[cid], scores[cid]) for cid in order]


def search_hybrid(question, top_k=5, pool=POOL, k=K):
    """混合检索：向量 + BM25 各取 pool 条 → RRF 融合 → 返回 top_k。接口与其他检索器一致。"""
    va = search_chroma(question, pool)
    bm = search_bm25(question, pool)
    return rrf_fuse([va, bm], k)[:top_k]


if __name__ == "__main__":
    for q in ["打车费能报销吗？", "公司 VPN 怎么用？", "公司有健身房吗？"]:
        print("=" * 72)
        print("问：", q)
        print("  【RRF 融合后 top5】")
        for c, s in search_hybrid(q, 5):
            print(f"   {s:.5f}  《{c['doc_name']}》{c['section']}")
    




















