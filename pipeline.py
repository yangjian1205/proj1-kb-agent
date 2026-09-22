# pipeline.py
# 知识库问答的编排层：检索 → 门卫分流 → 生成 → 打包结果
# 为什么单独抽一层：命令行（main.py）和 HTTP（app.py）共用这一份逻辑，
# 换入口不用改核心 —— 这就是「核心逻辑与交互方式解耦」。


import sys,os,time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)

from kb.guard import retrieve,route,REFUSE_REPLY,HARD_REFUSE,PASS
from llm.generator import generate,can_answer

# 拒答的单一出口
def _refuse(question,tier,top_sim,started,llm_calls,reason=""):  # _refuse 内部使用外部别直接调
    """所有拒答路径统一走这里 —— 保证「拒答」和「正常回答」返回的字段完全一致。

    为什么非要一致：调用方（前端/别的程序）不用写 if 判断"这次返回的是哪种结构"，
    少一堆分支就少一堆 bug。（下划线开头表示"内部函数"，外面不直接用）
    """
    return {
        "question": question,
        "answer": REFUSE_REPLY,  # 统一话术 知识库未找到相关信息
        "refused": True,     # 判断是否拒答
        "tier": tier,       # 那几个档位
        "top_sim": round(top_sim, 4),  # 保留4位小数
        "citations": [],
        "check": {"used": [], "invalid": [], "coverage": False},  
        "llm_calls": llm_calls,
        "latency_ms": int((time.perf_counter() - started) * 1000), # 当前秒数-走过的秒数 转换成毫秒 去掉小数保留整数
        "judge_reason": reason,
    }

def answer_question(question: str,top_k: int = 5)-> dict:  # 返回字典
    """完整问答链路，返回结构化结果（全程不打印任何东西）。

    tier 三个取值（来自 kb/guard.py 的 route()）：
      refuse —— 低于硬拒线，直接拒答，一次 LLM 都不调
      judge  —— 灰区，先让 LLM 判断「这些资料够不够回答」
      answer —— 高于放行线，直接生成
    """
    started = time.perf_counter()     # 计时起点
    llm_calls = 0    # 数一下调用了几次模型接口

    hits,top_sim = retrieve(question,top_K=top_k)
    tier = route(top_sim)

    
    # ---- 第一层：检索层拒答（0 次 LLM 调用，省钱）----
    if tier == "refuse":
        return _refuse(question,tier,top_sim,started,llm_calls)

    # ---- 第二层：灰区判定（花 1 次 LLM）----
    # 调用一次llm看看是否能回答
    judge_reason = ""  # 预制空字符串，只有下面的return才会用到
    if tier == "judge":
        j = can_answer(question,hits)   # 他的返回值是{'can_answer': bool, 'reason': str}
        llm_calls += 1  # 这是花了钱的
        judge_reason = j["reason"]  # 从裁判返回的字典里取出理由字符串，比如 '资料未提及公司是否有健身房'。
        if not j["can_answer"]:
            return _refuse(question,tier,top_sim,started,llm_calls,judge_reason)

    # ---- 第三层：生成带引用的答案（花 1 次 LLM）----
    out = generate(question, hits)  
    llm_calls += 1

    return {
        "question": question,
        "answer": out["answer"],
        "refused": False,
        "tier": tier,
        "top_sim": round(top_sim, 4),
        "citations": out["citations"],
        "check": out["check"],
        "llm_calls": llm_calls,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "judge_reason": judge_reason,
    }

if __name__ == "__main__":
    # 自测：三道题把三个档位和两种拒答路径都走一遍
    for q in ["报销票据有什么要求？",      # 预期 answer
              "公司有健身房吗？",             # 预期 judge → 拒答
              "公司股票今天多少钱一股？"]:    # 预期 refuse
        r = answer_question(q)
        print("=" * 70)
        print(f"问：{r['question']}")
        print(f"档位={r['tier']}  相似度={r['top_sim']}  拒答={r['refused']}"
              f"  耗时={r['latency_ms']}ms  LLM调用={r['llm_calls']}次")
        print("答：", r["answer"][:80]) # 这里有引用[1]就是chunks id 1





