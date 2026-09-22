# llm/generator.py
# 生成器：把检索到的资料变成带编号引用的答案
# 三个函数分工明确：
#   generate()        —— 生成答案 + 打包引用清单 + 核验引用
#   can_answer()      —— 灰区判定：这些资料够不够回答问题（只回 JSON，不生成答案）
#   check_citations() —— 从答案里抠出 [N] 编号，检查有没有越界编造
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, re, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-chat"
MAX_CHARS = 1000   # 每条资料截断长度：5 条 × 1000 字，够用且不超上下文

SYSTEM = """你是公司内部制度问答助手。严格遵守以下规则：

【回答规则】
1. 只能依据【参考资料】作答。资料里没写的一律不说，禁止用常识或经验补全。
2. 先给结论（一句话），再补细节（数字、条件、例外）。
3. 每句包含事实的话后面必须标注来源编号，格式 [1]；一句话依据多份资料就写 [1][3]。
4. 资料不足以回答问题时，只回复「知识库中未找到相关信息。」—— 不解释、不猜测、不给建议。

【引用规则】
- 编号只能取自【参考资料】里已有的编号，严禁自己编造。
- 只标注你这句话真正依据的那几条编号。"""

JUDGE_SYSTEM = """你是检索质量审核员。用户提了一个问题，系统检索到若干参考资料片段。
你的唯一任务：判断这些资料是否足以回答该问题。

判断规则：
1. 依据是「资料里有没有问题所问的那个事实」，不是「资料主题是否沾边」。
2. 只要资料里明确写出了答案（哪怕用词不同、需要简单推理），就算 can_answer=true。
3. 资料里只提到相关领域但没写出问题所问的事实（例：问公司有没有健身房，资料只提体检福利），算 can_answer=false。
4. 拿不准时倾向 false —— 宁可说不知道，也不能编。

只输出一行 JSON：{"can_answer": true 或 false, "reason": "不超过 20 字的中文理由"}"""


def build_context(hits):
    """把 [(块, 分数)] 拼成带 [1][2] 编号的参考资料文本"""
    return "\n\n".join(
        f"[{i + 1}] 《{c['doc_name']}》{c['section']}\n{c['text'][:MAX_CHARS]}"
        for i, (c, _s) in enumerate(hits)
    )
#  抬头和正文第一行有点重复。看 [1]：抬头已经写了《财务报销管理制度》第二章，正文第一行又是 财务报销管理制度 · 第二章 报销票据要求 —— 因为 chunk 正文里本来就把"文档名 · 章节名"当首行存了（这是当初切块时留下的）。
def build_citations(hits):  # 造引用
    """生成引用清单：每条资料的结构化出处，供前端做锚点跳转"""
    out = []
    for i,(c,s) in enumerate(hits,start=1):
        out.append({
            "no": i,
            "chunk_id": c["chunk_id"],
            "doc_name": c["doc_name"],
            "section": c["section"],
            "source": c["source"],
        })
    return out

def check_citations(answer, n_hits):
    """核验答案里的 [N] 编号：用到哪些、有没有越界编造、一个都没标吗"""
    used = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    # 捕获编号转数字去重，在排序
    invalid = [x for x in used if x < 1 or x > n_hits]
    # 筛选不存在的 从1开始也就是说不会有小于1的  一共5条不会有大于5的
    return {"used": used, "invalid": invalid, "coverage": bool(used)}
    # 引用的编号 不存在的编号 有没有引用如果FLASE大概率是瞎编的

def generate(question, hits):
    """生成带引用的答案。返回 dict，不返回裸字符串 —— 因为调用方需要引用清单和核验结果"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"【参考资料】\n{build_context(hits)}\n\n【问题】{question}"},
        ],
        temperature=0.2,          # 低温度 = 少自由发挥，尽量贴着资料答
    )
    answer = resp.choices[0].message.content
    return {
        "answer": answer, # 模型生成
        "citations": build_citations(hits), # 上的返回值列表套字典
        "check": check_citations(answer, len(hits)), # 上边的返回值
    }

def can_answer(question, hits):
    """灰区判定：这些资料够不够回答问题？返回 {'can_answer': bool, 'reason': str}"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",
             "content": f"【问题】{question}\n\n【参考资料】\n{build_context(hits)}"},
        ],
        temperature=0,              # 判定要稳定，温度压到 0
        response_format={"type": "json_object"},   # 强制输出合法 JSON，省去解析兜底
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw) # 字符串 → Python 字典。因为第 ⑦ 行加了约束，这里拿到的基本一定是合法的。
        return {"can_answer": bool(data.get("can_answer")), # data = json.loads(raw)
                                                            # 值  : {'can_answer': False, 'reason': '资料未提及公司是否有健身房'}
                                                            # type: dict

                "reason": str(data.get("reason", ""))}
    except json.JSONDecodeError:
        return {"can_answer": False, "reason": f"解析失败，按不可答处理：{raw[:40]}"}
# 灰色判定区域，手里这些资料够不够回答用户问题，够就try 不够就expect

if __name__ == "__main__":
    # 自测：走一遍「检索 → 生成 → 核验」的完整链路
    from kb.guard import retrieve
    q = "报销票据有什么要求？"
    hits, sim = retrieve(q)
    out = generate(q, hits)
    print(out["answer"])
    print("\n--- 引用核验 ---")
    print(out["check"])






























