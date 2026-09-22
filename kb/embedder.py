# 批量向量化
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 项目根目录
sys.path.insert(0, str(ROOT))                    # 让 import kb 找得到
os.chdir(ROOT)                                   # 把工作目录切到项目根

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "text-embedding-v3"
BATCH = 10   # 通过这批embedding 接口次数有限制，10条一批最稳

def embed_texts(texts,batch_size=BATCH):
    vecs = []
    for i in range(0,len(texts),batch_size): # 起点，终点，步长
        batch = texts[i:i+batch_size]  # 不要一次性取太多会报错，这里算是一个保险
        resp = client.embeddings.create(model=MODEL,input=batch) # 调用模型向量化
        vecs.extend([d.embedding for d in resp.data])  # 把向量化的结果拆开装进vecs,这里不是一口气全塞进去
        print(f"已经向量化{len(vecs)}/{len(texts)}")
    return np.array(vecs,dtype="float32")   # 转成numpy 矩阵
def embed_one(text):
    resp = client.embeddings.create(model=MODEL, input=[text]) # 列表，只接收1条信息
    return np.array(resp.data[0].embedding, dtype="float32")  # 所以这里只取第一个，一维向量，一条
# 这段代码是用户查询用的
if __name__ == "__main__":
    import json
    chunks = json.load(open("kb/chunks.json", encoding="utf-8")) # 解析成Python能用的格式，上一步生成的
    vectors = embed_texts([c["text"] for c in chunks]) # 这里就是批量向量化 二维矩阵
    np.save("kb/vectors.npy", vectors) # 把 vectors 这个矩阵存盘，存成 numpy 专用的 .npy 文件
    print(f"完成：{vectors.shape[0]} 条向量，每条 {vectors.shape[1]} 维 → kb/vectors.npy")
    # 行数 跟列数（维数）


















