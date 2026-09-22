# scripts/eval_guard.py
# D24 拒答评测：50 题全跑，看门卫拦得对不对
# 三条指标：库外召回 / 误拒率 / 灰区触发率
# 用法：python scripts/eval_guard.py            # 离线：只用 D23 缓存里的向量分扫阈值（零接口成本）
#       python scripts/eval_guard.py --online   # 在线：真跑 50 题检索 + 灰区题调 LLM 判定



import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from kb.guard import retrieve, route, HARD_REFUSE, PASS
from llm.generator import can_answer

ONLINE = "--online" in sys.argv
CACHE  = "eval/_run_cache.json"       # D23 留下的三路结果缓存
OUT_MD = "notes/d24_guard_report.md"

# 阈值扫描点：从松到紧，覆盖 0.50~0.70，0.63 附近多放两个点（那是实测的甜点区）
THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.63, 0.65, 0.68, 0.70]

DATA = json.load(open("eval/questions.json", encoding="utf-8"))["questions"]

def load_cache():
    """从 D23 缓存取每题的「向量 top1 相似度」—— 零接口成本，专门用来快速调阈值。

    缓存结构：{题号: {"chroma": [[chunk_id, doc_name, section, 分数], ...], ...}}
    我们要的是 chroma 列表里第 1 条的最后一个字段。"""

    if not Path(CACHE).exists():
        print(f"找不到 {CACHE},离线模式不可用。先跑一次D23的 evaluate.py 生成缓存。")
        sys.exit(1)
    raw = json.load(open(CACHE,encoding="utf-8"))
    return {int(k): v["chroma"][0][3] for k, v in raw.items()}
# 一句话：这个函数不去调任何 API，而是把 D23 评测时已经存下来的缓存文件读进来，从每题那一大坨结果里，只挑出一个数字 —— "向量检索排名第 1 那条的相似度"，最后交出一张 {题号: 分数} 的对照表。

def sweep(sims):
    """阈值扫描（纯分数口径，不调 LLM）：每个阈值拦下多少库外、误杀多少可答"""
    rows = []
    for t in THRESHOLDS:  # 自定义的阈值表 下边的t一轮换一个
        caught = sum(1 for it in DATA if it["type"] =="ood" and sims[it["id"]]< t)
        killed = sum(1 for it in DATA if it["type"] != "ood" and sims[it["id"]] < t)
        rows.append((t,caught,killed))
    return rows
# 看拦截了多少，误杀了多少

def rows_from_cache(sims):
    """离线模式：分数直接从缓存读，灰区不给 LLM 判定（verdict 留 None）"""
    return [{"id": it["id"], "type": it["type"], "q": it["q"],
             "sim": sims[it["id"]], "tier": route(sims[it["id"]]),
             "verdict": None, "reason": ""}
            for it in DATA]
# 返回值列表套字典
# sim是缓存比如 向量 top1 相似度，如 0.86607
# tire是那三个档位 route() 的返回值：refuse / judge / answer

def run_online():
    """在线模式：真跑50题检索拿真实分数，灰度题在调一次llm判定"""
    rows = []
    for n,it in enumerate(DATA,start=1):
        hits,sim = retrieve(it["q"])
        tier = route(sim)
        verdict,reason = None,""
        if tier == "judge":
            j = can_answer(it["q"],hits)
            verdict,reason = j["can_answer"],j["reason"]
            time.sleep(0.3)
        rows.append({"id": it["id"], "type": it["type"], "q": it["q"],
                     "sim": sim, "tier": tier,
                     "verdict": verdict, "reason": reason})   
        flag = "" if verdict is None else f"  LLM={verdict}"
        print(f"[{n:>2}/50] #{it['id']:<2} {it['type']:<7} "
              f"sim={sim:.3f} → {tier}{flag}")
        time.sleep(0.15)                              # 限速，避免 429
    return rows

# 把50道题真实跑一遍，灰色区域才调用llm


def refused(r):
    """这一题最终会不会被拒答（把三档的结论统一成一个布尔值）"""
    if r["tier"] == "refuse":
        return True
    if r["tier"] == "judge":
        return r["verdict"] is False    # 离线模式下 verdict=None，不算拒答
    return False

def metrics(rows):
    ood = [r for r in rows if r["type"] == "ood"] # 5
    ans = [r for r in rows if r["type"] != "ood"] # 45
    return {
        "ood_total": len(ood),
        "ood_refused": sum(1 for r in ood if refused(r)),
        "ans_total": len(ans),
        "ans_refused": sum(1 for r in ans if refused(r)),  # 两组各拒了多少
        "judge_n": sum(1 for r in rows if r["tier"] == "judge"),  # 进灰区的
        "judge_calls": sum(1 for r in rows if r["verdict"] is not None),  # 进灰区并调用了llm的 调用了几次因为线上他不是None
    }
# 是算总账的函数——把 50 题跑完的结果按「库外 vs 可答」两堆分开，数出每堆里有多少被拒答，再顺手记一下灰区触发了多少次、真调 LLM 调了几次，最后返回一个装着 6 个数字的字典。

def write_report(rows, sw, m):
    mode = "在线（真跑 50 题检索）" if ONLINE else "离线（读 D23 缓存，灰区未判）"
    L = []
    L.append("# D24 · 拒答评测报告\n")
    L.append(f"日期：2026-09-17 ｜ 评测集：eval/questions.json（50 题 = 45 可答 + 5 库外）")
    L.append(f"判定：向量 top1 相似度 → 三档路由 ｜ 阈值：硬拒 {HARD_REFUSE} / 放行 {PASS} ｜ 模式：{mode}\n")

    L.append("## 一、阈值扫描（纯分数口径，不含 LLM 判断）\n")
    L.append("| 阈值 | 拦住库外 | 误杀可答 |\n|---|---|---|")
    for t, c, k in sw:
        L.append(f"| {t:.2f} | {c} / 5 | {k} / 45 |")   # .2f：让 0.50 显示成两位小数，别变成 0.5  这里t是阈值 c是题外 k是题内
    L.append("")

    L.append("## 二、三档分布\n")
    L.append("| 档位 | 题数 | 其中库外 | 其中可答 |\n|---|---|---|---|")
    # 档位名里不要用裸的 < 号：Markdown 表格里的 < 可能被渲染器当成 HTML 标签吞掉
    for tier, label in [("refuse", "硬拒（低于 0.55）"),
                          ("judge", "灰区（0.55~0.63）"),
                          ("answer", "放行（0.63 及以上）")]:
        sub = [r for r in rows if r["tier"] == tier]   # 把属于这一档的题目捞出来  sub 是 subset（子集）的缩写
        L.append(f"| {label} | {len(sub)} | "
                 f"{sum(1 for r in sub if r['type'] == 'ood')} | "
                 f"{sum(1 for r in sub if r['type'] != 'ood')} |")   # 列表已经建好 提数 其中库外 其中可达
    L.append("")

    L.append("## 三、灰区逐题（唯一需要 LLM 判断的一批）\n")
    L.append("| # | 类型 | 问题 | 向量分 | 档位 | LLM 判定 | 理由 | 真值 | 对错 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["tier"] != "judge":
            continue  # 不是灰区直接跳过
        truth = "库外" if r["type"] == "ood" else "可答"
        want = False if r["type"] == "ood" else True
        mark = "?" if r["verdict"] is None else ("✓" if r["verdict"] == want else "✗") # 三选一对错问号，条件2成立是灰区且want==F
        L.append(f"| {r['id']} | {r['type']} | {r['q']} | {r['sim']:.3f} | {r['tier']} | "
                 f"{r['verdict']} | {r['reason']} | {truth} | {mark} |")
    L.append("")

    # 灰区里 LLM 判对了几个：期望值 = 库外题判 False、可答题判 True
    judged = [r for r in rows if r["verdict"] is not None]
    judge_ok = sum(1 for r in judged if r["verdict"] == (r["type"] != "ood"))

    L.append("## 四、端到端结果\n")
    L.append(f"- 库外召回（该拒的都拒了吗）：**{m['ood_refused']} / {m['ood_total']}**")
    L.append(f"- 误拒率（不该拒的误拒了几个）：**{m['ans_refused']} / {m['ans_total']}**")
    L.append(f"- 灰区触发：**{m['judge_n']} / {len(rows)}** 题进了灰区，"
             f"实际调 LLM 判定 **{m['judge_calls']}** 次")
    L.append(f"- LLM 判断准确率（灰区内）：**{judge_ok} / {m['judge_calls']}**\n")

    L.append("## 五、结论（自己填）\n")
    L.append("- 三档路由相比单阈值（0.63）好在哪：___")
    L.append("- 灰区里 LLM 判断失手的是哪几题、为什么：___")
    L.append("- 这套方案的成本（每题平均几次接口调用）：___")
    L.append("- 如果语料扩到 1000 份文档，这个阈值还成立吗、你会怎么重扫：___")

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))
    print(f"\n已写入 {OUT_MD}（结论区留空，自己填）")


def main():
    sims = load_cache()
    rows = run_online() if ONLINE else rows_from_cache(sims)
    sw = sweep(sims)
    m = metrics(rows)

    print("\n" + "=" * 60)
    print(f"库外召回 {m['ood_refused']}/{m['ood_total']} ｜ "
          f"误拒 {m['ans_refused']}/{m['ans_total']} ｜ "
          f"灰区 {m['judge_n']} 题（LLM 实际调用 {m['judge_calls']} 次）")
    write_report(rows, sw, m)


if __name__ == "__main__":
    main()



