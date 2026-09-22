# 检索效果评测：用 50 题评测集，把「向量 / BM25 / RRF 融合」三路放在同一把尺子下量
# 指标：hit@5（前 5 条里有没有正确答案的出处）+ MRR@5（正确答案排得越靠前分越高）
# 判定粒度两套都算：
#   块级（严格）—— top5 中任一块的 chunk_id 命中 gold_chunks
#   文档级（宽松）—— 从 gold_chunks 反查文档名，top5 中任一块的 doc_name 命中即算
# 用法：python scripts/evaluate.py            # 实跑（会调 50 次 embedding）
#       python scripts/evaluate.py --offline  # 复用 eval/_run_cache.json，不调接口（改判定规则时用）

import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from kb.retriever_chroma import search_chroma
from kb.retriever_bm25 import search_bm25
from kb.fusion import rrf_fuse, POOL

K = 5                                            # 评测截断位：只看前 5 条
OFFLINE = "--offline" in sys.argv
CACHE = "eval/_run_cache.json"   # 纯字符串，路径
DATA = json.load(open("eval/questions.json", encoding="utf-8"))["questions"]  # 转成dict 并提取问题
CHUNKS = {c["chunk_id"]: c for c in json.load(open("kb/chunks.json", encoding="utf-8"))}  # 这里纯列表转字典
PATHS = [("chroma", "向量 Chroma"), ("bm25", "关键词 BM25"), ("hybrid", "RRF 融合")]  


def pack(hits):
    """[(块, 分数)] → [[chunk_id, doc_name, section, 分数]]，只留前 K 条"""
    return [[c["chunk_id"], c["doc_name"], c["section"], round(s, 5)] for c, s in hits[:K]]


def run_all():
    if OFFLINE:  # 判断走缓存还是真跑 如果手敲了OFFLINE就走下面的return，剩下的全部跳过
        print(f"离线模式：读{CACHE}(不调emmbedding接口)")
        return {int(k):v for k,v in json.load(open(CACHE,encoding="utf-8")).items()}  
    print("实跑 50 题（约 30~60 秒，每题一次 embedding 调用）...")
    out = {}
    for it in DATA:
        va = search_chroma(it["q"],POOL)    # 向量路：取20条候选
        bm = search_bm25(it["q"],POOL)    # 关键词路：取20条候选
        hy = rrf_fuse([va,bm])[:K]       # 融合后取前5
        out[it["id"]] = {"chroma":pack(va),"bm25":pack(bm),"hybrid":pack(hy)}  # 这里的pack是上方定义过的
        time.sleep(0.2)      # 轻微限速，避免50次连续调用接口限流
        print(".",end="",flush=True)  # 进度条

    json.dump({str(k): v for k, v in out.items()}, open(CACHE, "w", encoding="utf-8"),  # 写进文件，上边的是读 保留中文，缩进1，
              ensure_ascii=False, indent=1) # 这里的dump他和str算是绑定出现 写进的像是就是键值对的形式，上边参考上方return
    print(f"\n结果已缓存到 {CACHE}（下次可加 --offline 免接口重算）")
    return out       

def gold_docs_of(gcs):
    return {CHUNKS[i]["doc_name"] for i in gcs} # 那不就返回切片序号和大标题

def rank_of(hits,item,level):
    """返回第一条命中的排名（1 开始）；没命中返回 0。level: chunk / doc"""
    gcs = item["gold_chunks"]
    if not gcs:
        return 0                                 # 库外题没有 gold，不参与命中判定
    if level == "chunk":
        target, key = set(gcs), lambda cid: cid
    else:
        target, key = gold_docs_of(gcs), lambda cid: CHUNKS[cid]["doc_name"]   # 自动反查文档名
    for rank, row in enumerate(hits[:K], start=1):
        if key(row[0]) in target:
            return rank
    return 0

# [ row[0],   row[1],    row[2],    row[3] ]
#   chunk_id  doc_name   section    分数
 
def metrics(res, level, only_type=None):
    """算一组题在三条路上的 (hit@5, MRR@5)；only_type 给定时只统计该题型"""
    sub = [it for it in DATA if it["type"] != "ood" and (only_type is None or it["type"] == only_type)]
    # 不相关的题型直接踢开，没有参数就全都要，或者只要传的类型的
    if not sub:
        return {},0
    row = {}
    # | 检索路 | 块级 hit@5 | 块级 MRR@5 | 文档级 hit@5 | ... |
    # | 向量 Chroma | 100.0% | 0.985 | ... |        ← 这就是一个 row
    for p, _ in PATHS:
        ranks = [rank_of(res[it["id"]][p], it, level) for it in sub]
        # res                              # run_all() 的返回：{题号: {路名: 结果}}
        # res[it["id"]]                    # 取出这一道题 → {"chroma": [...], "bm25": [...], "hybrid": [...]}
        # res[it["id"]][p]                 # 再取出当前这一路 → [[cid, doc, sec, score], ...]  ← 这就是 hits

        row[p] = (sum(1 for r in ranks if r > 0) / len(sub), # hit@5分子  
                  sum(1 / r if r else 0 for r in ranks) / len(sub))  # MRR@5分子
    return row, len(sub)

    # row = {
    #     "chroma": (1.000, 0.985),   # ← 这两个数就是 row[p]
    #     "bm25":   (0.933, 0.839),
    #     "hybrid": (0.978, 0.924),
    # }

def main():
    res = run_all()
    answered = [it for it in DATA if it["type"] != "ood"]  # 45
    ood = [it for it in DATA if it["type"] == "ood"]   # 5
    out = []

    # 一、总体（块级 + 文档级 一起报）
    # 这五行干的事：用"块级"和"文档级"两把尺子，把三条检索路的成绩各量一遍，然后按检索路一行一行写成 Markdown 表格。
    out.append(f"## 一、总体（{len(answered)} 道可答题）\n")
    out.append("| 检索路 | 块级 hit@5 | 块级 MRR@5 | 文档级 hit@5 | 文档级 MRR@5 |")
    out.append("|---|---|---|---|---|")
    chunk_m,_ = metrics(res,"chunk")
    #     {
    #   "chroma": (1.0, 0.985...),   # (hit@5, MRR@5)
    #   "bm25":   (0.9333..., 0.839...),
    #   "hybrid": (0.9777..., 0.924...),
    # }
    doc_m, _ = metrics(res, "doc") # 文档级跟块级
    for p, label in PATHS:  # p是英文标签那仨，label是中文标签美观
        out.append(f"| {label} | **{chunk_m[p][0]:.1%}** | **{chunk_m[p][1]:.3f}** | "
                   f"{doc_m[p][0]:.1%} | {doc_m[p][1]:.3f} |")
    # 取出 p 路的 (hit@5, MRR@5) 元组 [0]	元组第 1 个 → hit@5；[1] → MRR@5
    # :.1%	百分比格式、保留 1 位小数。0.9777 → 97.8%（会四舍五入，1.0 → 100.0%）
    # :.3f	定点小数、保留 3 位。0.8391... → 0.839

    # 二、分题型（块级口径）
    # 这 8 行干的事：加一个二级标题，然后把 5 种题型各跑一遍指标，每种题型写成一个表格行。表头固定 8 列，循环体负责填满这 8 格。
    out.append("\n## 二、分题型（块级判定）\n")
    out.append("| 题型 | 题数 | 向量 hit@5 | BM25 hit@5 | RRF hit@5 | 向量 MRR | BM25 MRR | RRF MRR |")
    out.append("|---|---|---|---|---|---|---|---|")
    for t in ["direct", "synonym", "entity", "numeric", "multi"]:
        m, n = metrics(res, "chunk", t)
        out.append(f"| {t} | {n} | {m['chroma'][0]:.0%} | {m['bm25'][0]:.0%} | {m['hybrid'][0]:.0%} | "
                   f"{m['chroma'][1]:.3f} | {m['bm25'][1]:.3f} | {m['hybrid'][1]:.3f} |")
    # .0% 百分比格式保留0位小数


    # 三、逐题明细
    # 这段是报告第三节的生成器 —— 把 45 道可答题一题一行铺成 9 列的 Markdown 表，前 3 列说是哪道题，中间 3 列说正确那一段在三路里排第几，后 3 列说三路各自的第一名来自哪份文档。
    out.append("\n## 三、逐题明细（块级排名 = 正确那一段排第几，0 = 前 5 条里没有）\n")
    out.append("| # | 题型 | 问题 | 向量 | BM25 | RRF | 向量 top1 | BM25 top1 | RRF top1 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for it in answered:  # 这里不写DATA是因为他有50个没筛选过的
        r = res[it["id"]]
        out.append(f"| {it['id']} | {it['type']} | {it['q']} | "   
                   f"{rank_of(r['chroma'], it, 'chunk') or '✗'} | "  # 返回1—5 表示命中 和上面的r是同一个变量只不过这里单取其中一个
                   f"{rank_of(r['bm25'], it, 'chunk') or '✗'} | "
                   f"{rank_of(r['hybrid'], it, 'chunk') or '✗'} | "
                   f"{r['chroma'][0][1]} | {r['bm25'][0][1]} | {r['hybrid'][0][1]} |")
                    #         {
                    #   'chroma': [
                    #     [1,   '财务报销管理制度',   '第二章 报销票据要求', 0.86607],
                    #     [180, '财务报销FAQ',        '一、发票与票据',     0.76161],
                    #     [0,   '财务报销管理制度',   '第一章 总则',        0.7368 ],
                    #     [125, '费用报销操作指南',   '一、报销前准备',     0.73304],
                    #     [126, '费用报销操作指南',   '二、OA 提交步骤',    0.72871]
                    #   ],
                    #   'bm25': [
                    #     [1,   '财务报销管理制度',   '第二章 报销票据要求', 10.64994],
                    #     [3,   '财务报销管理制度',   '第四章 报销流程',     7.16618],
                    #     [202, '办公生活FAQ',        '六、差旅生活',        7.04975],
                    #     [180, '财务报销FAQ',        '一、发票与票据',      7.02716],
                    #     [32,  '差旅管理制度',       '第六章 差旅费用报销', 6.58683]
                    #   ],
                    #   'hybrid': [
                    #     [1,   '财务报销管理制度',   '第二章 报销票据要求', 0.03279],
                    #     [180, '财务报销FAQ',        '一、发票与票据',      0.03175],
                    #     [3,   '财务报销管理制度',   '第四章 报销流程',      0.03128],
                    #     [0,   '财务报销管理制度',   '第一章 总则',         0.03102],
                    #     [125, '费用报销操作指南',   '一、报销前准备',      0.03055]
                    #   ]
                    # }

        # | 11 | synonym | 年假有几天？ | 3 | ✗ | 1 | 员工手册 | 财务报销管理制度 | 员工手册 |

    # 四、融合的得与失
    # 这段是给「融合」做绩效评估：从 45 题里挑出两类极端情况记账 —— 融合救回了哪几题、融合搞砸了哪几题。整个报告就这两行最有说服力，因为它是证据，不是百分比。
    saved, hurt = [], []
    for it in answered:
        r = res[it["id"]]
        rc = rank_of(r["chroma"], it, "chunk")
        rb = rank_of(r["bm25"], it, "chunk")
        rh = rank_of(r["hybrid"], it, "chunk")
        if rh > 0 and rc == 0 and rb == 0:
            saved.append(it["id"])  # 三者同时成立才算
        if rh == 0 and (rc > 0 or rb > 0):
            hurt.append(it["id"]) # 这里也是同理
    out.append("\n## 四、融合的得与失（块级口径）\n")
    out.append(f"- **融合救回**（两路单跑都 ✗、融合后 ✔）的题：{saved or '无'}")
    out.append(f"- **融合拖累**（某单路 ✔、融合后 ✗）的题：{hurt or '无'}")

    # 五、库外题 + 拒答阈值区间
    # 这段是第五节——把 5 道库外题列出来（它们本该答不上来），再拿"可答题的分数区间"和"库外题的分数区间"摆在一起对照，好让你定出拒答阈值；最后把所有攒好的行拼成一篇报告打印出来。
    out.append("\n## 五、库外题（正确行为 = 不知道就说不知道）\n")
    ood_rows = [(it, res[it["id"]]) for it in ood if it["id"] in res]
    if not ood_rows:
        out.append("（当前缓存里没有库外题数据，本节跳过 —— 跑一次实跑模式即可补上）")
    else:
        out.append("| # | 问题 | 向量最高相似度 | BM25 最高分 | 向量 top1 | BM25 top1 |")
        out.append("|---|---|---|---|---|---|")
        for it, r in ood_rows:
            out.append(f"| {it['id']} | {it['q']} | {r['chroma'][0][3]:.3f} | {r['bm25'][0][3]:.2f} | "
                       f"{r['chroma'][0][1]} | {r['bm25'][0][1]} |")

    ok_rows = [res[it["id"]] for it in answered if it["id"] in res]
    ok_sim = [r["chroma"][0][3] for r in ok_rows]
    ok_bm = [r["bm25"][0][3] for r in ok_rows]
    out.append("\n**定拒答阈值用的区间对照**")
    out.append(f"- 可答题 · 向量最高相似度：min {min(ok_sim):.3f} / max {max(ok_sim):.3f}")
    out.append(f"- 可答题 · BM25 最高分：min {min(ok_bm):.2f} / max {max(ok_bm):.2f}")
    if ood_rows:
        ood_sim = [r["chroma"][0][3] for _, r in ood_rows]
        ood_bm = [r["bm25"][0][3] for _, r in ood_rows]
        out.append(f"- 库外题 · 向量最高相似度：{', '.join(f'{x:.3f}' for x in ood_sim)}")
        out.append(f"- 库外题 · BM25 最高分：{', '.join(f'{x:.2f}' for x in ood_bm)}")
        # 把五个分数转成字符串，并保留三位小数
    report = "\n".join(out)
    print("\n" + report)

    # 顺手落盘成笔记，结论区留空

    Path("notes").mkdir(exist_ok=True)
    with open("notes/d23_eval_report.md", "w", encoding="utf-8") as f:
        f.write("# D23 · 检索评测报告\n\n")
        f.write("日期：2026-09-15 ｜ 评测集：eval/questions.json（50 题 = 45 可答 + 5 库外）\n")
        f.write("判定：块级（主）+ 文档级（对照）｜ 截断位：top5\n\n")
        f.write(report)
        f.write("\n\n## 六、结论（自己填）\n\n")
        f.write("- 块级判定下，RRF 相比最强单路：hit@5 ___ → ___，MRR ___ → ___\n")
        f.write("- 融合救回了哪几题：___；融合拖累了哪几题：___\n")
        f.write("- 为什么文档级指标三路全是 100%：___\n")
        f.write("- 拒答阈值：向量相似度 < ___ 或 BM25 < ___ 判定「库外 / 没找到」\n")
    print("\n已写入 notes/d23_eval_report.md（结论区留空，自己填）")


if __name__ == "__main__":
    main()