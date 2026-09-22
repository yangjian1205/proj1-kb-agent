# 拒答门卫：决定「这个问题该不该答」，不负责「答什么」
# 为什么需要三档而不是一个阈值：D23 实测发现向量相似度的两个区间重叠
#   —— 可答题最低 0.605，库外题最高 0.626，一根线切下去必然误判。
# 三档的分工：
#   < HARD_REFUSE          → 直接拒答（不调 LLM，省钱）
#   HARD_REFUSE ~ PASS     → 灰区，交 LLM 判断「资料够不够回答」
#   >= PASS               → 放行，直接生成答案


import sys,os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)

from kb.retriever_chroma import search_chroma
from kb.retriever_bm25 import search_bm25
from kb.fusion import rrf_fuse,POOL

# ---- 阈值：这两个数是在 eval/questions.json 的 50 题上扫出来的，换语料要重扫 ----
HARD_REFUSE = 0.55   # 低于此值：判为库外，直接拒答（实测误杀 0 道可答题）
PASS        = 0.63   # 高于此值：判为库内，直接回答（实测漏放 0 道库外题）

# 统一的拒答话术：所有拒答路径都返回这一句，前端不用分辨"是哪种拒答"
REFUSE_REPLY = "知识库中未找到相关信息。"

def route(top_sim):
    """按向量最高相似度返回档位：'refuse' / 'judge' / 'answer'"""
    if top_sim < HARD_REFUSE:
        return "refuse"
    if top_sim >= PASS:
        return "answer"
    return "judge"        # 灰区：两条线之间

def retrieve(question,top_K=5):
    """一次检索，两个用途：融合结果给生成用，向量 top1 分数给门卫用。

    返回 (hits, top_sim)：
      hits    —— [(块, rrf分), ...]，长度为 top_k
      top_sim —— 向量路第 1 条的余弦相似度，喂给 route()
    """
    va = search_chroma(question,POOL)  
    bm = search_bm25(question,POOL)
    hits = rrf_fuse([va,bm])[:top_K]
    top_sim = va[0][1] if va else 0.0
    return hits,top_sim
# va = [
#     ( {chunk_id:76, doc_name:"财务报销管理制度", text:"..."} , 0.812 ),
#     ( {chunk_id:211, ...} , 0.795 ),
#     ...
# ]   ← 按相似度从高到低排（Chroma 的 query 默认距离升序 = 相似度降序）


if __name__ == "__main__":
    # 三道题正好覆盖三个档位，跑完应该看到 refuse / judge / answer 各一次
    for q in ["公司股票今天多少钱一股？",     # 库外，预期 refuse
              "公司有健身房吗？",               # 库外，预期 judge
              "报销票据有什么要求？"]:          # 可答，预期 answer
        hits, sim = retrieve(q)
        print("=" * 72)
        print("问：", q)
        print(f"  向量 top1 相似度 = {sim:.4f}  →  档位：{route(sim)}")
        for c, s in hits[:3]:
            print(f"   {s:.5f}  《{c['doc_name']}》{c['section']}")














