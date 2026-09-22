# main.py
# 命令行入口：只负责「把结果打印得好看」，业务逻辑全在 pipeline.py
from pipeline import answer_question, HARD_REFUSE, PASS


def show(question):
    r = answer_question(question)

    print("\n" + "=" * 66)
    print("问：", r["question"])
    print(f"【检索】向量最高相似度 = {r['top_sim']:.4f}"
          f"（硬拒线 {HARD_REFUSE} / 放行线 {PASS}）")

    if r["tier"] == "refuse":
        print("【决策】低于硬拒线 → 直接拒答（不调 LLM，省一次调用）")
        print(f"\n答：{r['answer']}")
        return

    if r["tier"] == "judge":
        print(f"【决策】落在灰区 → LLM 判定：{r['judge_reason']}")
        if r["refused"]:
            print(f"\n答：{r['answer']}")
            return
    else:
        print("【决策】高于放行线 → 直接回答")

    print("\n答：", r["answer"])

    print("\n--- 引用来源 ---")
    for c in r["citations"]:
        print(f" [{c['no']}] 《{c['doc_name']}》{c['section']}")
        print(f"     {c['source']}")

    ck = r["check"]
    if not ck["coverage"]:
        print("\n⚠️ 引用核验：答案里没有出现任何 [N] 标记")
    elif ck["invalid"]:
        print(f"\n⚠️ 引用核验：出现越界编号 {ck['invalid']}（幻觉引用）")
    else:
        print(f"\n✅ 引用核验：用到 {ck['used']}，全部有效")

    print(f"\n[耗时 {r['latency_ms']} ms · 接口调用 {r['llm_calls']} 次]")


if __name__ == "__main__":
    print("知识库就绪（混合检索 + 拒答门卫 + 引用溯源）！"
          "输入问题回车提问，输入 exit 退出。\n")
    while True:
        q = input("你问： ").strip()
        if q.lower() in ("exit", "quit", "退出"):
            break
        if q:
            show(q)