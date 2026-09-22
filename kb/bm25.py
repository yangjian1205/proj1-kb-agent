# 构建索引 + 打分

import sys,os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import json
import math
from collections import Counter
import jieba

k1 = 1.5 # 词频饱和参数：越大，"重复出现的收益越持久"
B = 0.75  # 长度归一化：0=完全不惩罚长块 1=完全按比例惩罚

PUNCT = set("，。、；：？！（）《》【】“”‘’·—,.;:?()[]!\"' \n\t")
# 一对符号查询在不在里面很快

def tokenize(text):
    """中文分词 + 去噪：返回列表"""
    words = []
    for w in jieba.lcut(text.lower()): # 切分
        w = w.strip() # 去空白
        if not w or all(ch in PUNCT for ch in w): # W里的字符在不在PUNCT里
            continue
        # 单字中文（"的""了""元"）噪声大，丢掉；英文/数字（vpn、22）无论长短都留
        if len(w) >= 2 or w.isascii(): # 是否由字符组成
            words.append(w)
    return words

class BM25:
    def __init__(self,chunks):
        self.chunks = chunks
        self.docs = [tokenize(c["text"]) for c in chunks] # 每块的词列表
        self.tf = [Counter(d) for d in self.docs]   # 每块：词 出现的次数
        self.length = [len(d) for d in self.docs]   # 每块的词数
        self.N = len(chunks)
        self.avgdl = sum(self.length) / self.N

        # 预先算好每个词的IDF 只跟“多少块含这个词有关”，与查询无关，所以一次算完复用

        df = Counter()
        for d in self.docs:
            df.update(set(d))   # 这里是去重
        self.idf = {
            w: math.log(1 + (self.N - n +0.5) / (n + 0.5))
            for w,n in df.items()
        }
        # 在这里算词频逆文档是因为 他只跟语料有关 跟查询次数无关

    def search(self,query,top_k=5):
        words = [w for w in tokenize(query) if w in self.idf]  # 库里没有的词直接丢，别浪费算力
        scores = [0.0] * self.N  # 没有就全0分
        for w in words:
            idf = self.idf[w]
            for i in range(self.N):
                f = self.tf[i].get(w,0)   # 取出出现次数 没有就返回0
                if not f:
                    continue       # 没有这个词不加分

                denom = f + k1 * (1-B + B * self.length[i] / self.avgdl)  # 词频 + 词频饱和参数*长度归一化因子
                scores[i] += idf * f * (k1 + 1) / denom
        order = sorted(range(self.N),key=lambda i:scores[i],reverse=True)[:top_k] # (range(self.N) 214 按分数排索引
        return [(self.chunks[i],scores[i]) for i in order]
    
if __name__ == "__main__":
    chunks = json.load(open("kb/chunks.json", encoding="utf-8"))
    idx = BM25(chunks)
    print(f"建索引完成：{idx.N} 块，平均 {idx.avgdl:.1f} 词/块，词典 {len(idx.idf)} 个词")
    for q in ["公司 VPN 怎么用", "打车费能报销吗", "公司有健身房吗"]:
        print("=" * 60)
        print("问：", q, "| 有效查询词：", [w for w in tokenize(q) if w in idx.idf])
        hits = idx.search(q, 5)  # query top_5
        for c, s in hits:  # 元祖 c拿chunks  s拿scores
            print(f"  {s:7.2f} | 《{c['doc_name']}》{c['section']}")
        if hits[0][1] < 2: # 分数小于2  对应昨天的代码
            print("  ⚠️ 最高分断崖式偏低 = 几乎没有词命中 → 应判定为「没找到」")       


        
# hits[0]     = (dict, float)     ← 就两格
# hits[0][0]  = chunk 字典（doc_name = 加班管理制度）
# hits[0][1]  = 7.627806735858393 ← 类型 float，这是【分数】


























