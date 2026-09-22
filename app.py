# app.py
# HTTP 入口：把 pipeline.answer_question 暴露成 REST 接口
# 启动：uvicorn app:app --reload --port 8000
# 文档：http://127.0.0.1:8000/docs


from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

from pipeline import answer_question

app = FastAPI(
    title="企业知识库问答API",
    version="1.0.0",  # 给人看的
    description="基于 RAG 的内部制度问答：混合检索（BM25 + 向量）+ RRF 融合"
                " + 三档拒答 + 引用溯源",
)

# ---------- 出入参模型 ----------

class AskRequest(BaseModel): # 定义类 集成三样东西 自动类型校验 自动类型转换 
    question: str = Field(...,min_length=1,max_length=200,  # ...表示没有默认值 最小长度拦住空字符串，最长拦住过长的问题稀释语义 且embedeing模型有token上限
                          description="用户问题",
                          examples=["报销票据有什么要求"])
    top_k: int = Field(5,ge=1,le=20,description="检索条数,默认5")
    # 检索池子最大就20 来自D23

    
class Citation(BaseModel):
    no: int
    chunk_id: int
    doc_name: str
    section: str
    source: str
    # 引用长啥样 这段的作用

class CheckResult(BaseModel):
    used: list[int]
    invalid: list[int]
    coverage: bool
    # 检查有没有引用，跟之前差不多
# 大体长这样   
# 你问一个问题
#    ↓
# retrieve() → hits（5 条候选块）
#    ↓
# generate(question, hits)            # generator.py:78
#    ├─ answer    = 模型输出，正文带 [1][3] 标记
#    ├─ citations = build_citations(hits)  → 5 条编号 1~5 的出处清单（"菜单"）
#    └─ check     = check_citations(answer, len(hits))  → 本次质检结果（"点了哪几号菜"）
#    ↓
# pipeline.answer_question() 把三个键原样带回
#    ↓
# app.py 里 AskResponse.check: CheckResult → FastAPI 按模型过滤 + 序列化成 JSON

class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool = Field(..., description="是否拒答")
    tier: str = Field(..., description="决策档位：refuse / judge / answer")
    top_sim: float = Field(..., description="向量路 top1 余弦相似度")  # 这三个都是必填加说明 且都没有默认值
    citations: list[Citation] # 引用清单，5 条候选出处
    check: CheckResult  # 引用质检回执
    llm_calls: int = Field(..., description="本次消耗的大模型调用次数")
    latency_ms: int # 端到端时间，花了多少毫秒
    judge_reason: str = ""  # 这里可不传 灰区判定理由


# ---------- 路由 ----------
@app.get("/health",summary="健康检查")
def health():
    return {"status":"ok"} # 他只证明进程还活着

@app.post("/ask",response_model=AskResponse,summary="知识库问答")  # 生成文档  出参过滤 + 校验 简单来说就是丢掉多余字段
def ask(req:AskRequest): # 内部读原始 body 字节流 → json.loads → AskRequest.model_validate(...) 他还是pydantic类型
    """注意这里是普通 def，不是 async def —— 原因见下面的讲解。"""
    try:
        return answer_question(req.question, top_k=req.top_k) # 正常就输出这个字典
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答链路异常：{e}") # 异常就走这条路

















