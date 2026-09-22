# BM25封装检索
# 作用：把 BM25 的接口也做成 search_bm25(question, top_k=5)，返回值格式跟 Chroma 版完全一样（[(块, 分数), ...]）。这样 D23 的融合代码只要改一个 import，两条路可以无缝互换 —— 接口统一是你能把代码写"长"下去的关键。


import sys,os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent # 相对路径变成绝对路径
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)

import json
from kb.bm25 import BM25

# 模块级只建一次索引：214块的分词大概1秒，之后每次查询都是毫秒级
CHUNKS = json.load(open("kb/chunks.json",encoding="utf-8"))
_INDEX = BM25(CHUNKS)

def search_bm25(question,top_K=5):
    return _INDEX.search(question,top_K)

if __name__ == "__main__":
    for q in ["公司VPN怎么用","门禁卡丢了怎么办","在职证明怎么开"]:
        print("=" * 60)
        print("问：", q)
        for c, s in search_bm25(q):
            print(f"  {s:7.2f} | 《{c['doc_name']}》{c['section']}")


















