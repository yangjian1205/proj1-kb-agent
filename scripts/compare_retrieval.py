# scripts/compare_retrieval.py（新建在 scripts/ 目录下）
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent      # scripts/ 的上一级 = 项目根
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from kb.retriever_chroma import search_chroma
from kb.retriever_bm25 import search_bm25

# (问题, 这题在考什么)
QUESTIONS = [
    ("打车费能报销吗？",          "基线题：看谁排得对"),
    ("加班到 22 点打车能报销吗？",   "数字/条款：考精确匹配"),
    ("公司 VPN 怎么用？",           "专有名词：BM25 主场"),
    ("在职证明怎么开？",           "口语 vs 文档词（用印）：向量主场"),
    ("门禁卡丢了怎么办？",         "专有名词 + 口语混合"),
    ("我出差回来怎么把钱要回来？",   "全口语提问：考分词运气"),
    ("公司食堂中午吃什么？",        "边界题：语料答「没有食堂+440 餐补」，不该拒答"),
    ("公司有健身房吗？",           "真·库外题：两路都该没结果 → 拒答"),
]

def key(c):                     # 用 chunk_id 当唯一标识，比较两路命中是否同一块
    return c.get("chunk_id")

for q, note in QUESTIONS:
    va = search_chroma(q, 5)
    bm = search_bm25(q, 5)
    set_a = {key(c) for c, _ in va}
    set_b = {key(c) for c, _ in bm}
    overlap = len(set_a & set_b)

    print("\n" + "=" * 72)
    print(f"问：{q}    ｜ 考点：{note}")
    print(f"两路 top5 重合：{overlap}/5")
    print("\n  【向量检索 Chroma】")
    for c, s in va:
        print(f"   {s:.3f}  《{c['doc_name']}》{c['section']}")
    print("  【关键词检索 BM25】")
    for c, s in bm:
        print(f"   {s:7.2f}  《{c['doc_name']}》{c['section']}")