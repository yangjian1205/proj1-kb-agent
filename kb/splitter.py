# 按照2级标题切分
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 项目根目录
sys.path.insert(0, str(ROOT))                    # 让 import kb 找得到
os.chdir(ROOT)                                   # 把工作目录切到项目根

import json,re
from kb.loader import load_docs

MIN_LEN = 20    # 太短的直接丢掉

def split_docs(docs):
    chunks,cid = [],0 # 装切出来的块 计数器
    for d in docs:
        # 在「行首的 ## 」前面切开：既兼容 "## 第三章 xxx" 也兼容 "## 一、xxx"
        for part in re.split(r"\n(?=##\s)",d["text"]): # 他能保留后面的##保证不被切掉
            lines = part.strip().split("\n") # 去掉空白和切成多段
            # ['## 第一章 总则', '', '1.1 为规范公司费用报销管理……', '', '1.2 本制度……', '', '1.3 费用报销……']
            head = lines[0].strip() if lines else ""
            # 算是一个兜底机制吧 有东西就取第一行并去掉首尾的空白否则就取空白
            if head.startswith("## "):
            # 如果以## 开头
                section = head.lstrip("# ").strip()
                # 从左边开始剥 0 1 2 在去掉有边的空白
                body = "\n".join(lines[1:]).strip()
            else:
                continue # 一级标题是一个#开头的
            if len(body) >= MIN_LEN:
                chunks.append({
                    "chunk_id": cid,
                    "doc_name":d["doc_name"],
                    "category":d["category"],
                    "section":section,
                    "source":d["path"],
                    "text": d["doc_name"] + " · " + section + "\n" + body,
                })
                cid += 1
    return chunks

if __name__ == "__main__":
    chunks = split_docs(load_docs()) # 把34分文档切成200多块，并返回列表
    out = "kb/chunks.json"     # 文件路径字符串
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"共切出 {len(chunks)} 块 → {out}")
    print("样例：", chunks[0]["doc_name"], "|", chunks[0]["section"])















