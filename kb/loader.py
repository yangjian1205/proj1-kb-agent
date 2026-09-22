# 加载语料库（那三个1文件夹）
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent.parent / "docs"
CATEGORY = {"policies":"制度","guides":"流程指南","faqs":"FAQ"}
# 绝对路径 再往上调两层 再后面加上docs
# 这三个是docs下的三个文件夹和中文名

def load_docs():
    """遍历 docs/ 三个子目录，返回[{'doc_name','category','path','text'},...]"""
    # 这里小写是代表三种文件之一 取决于在哪里
    docs = []
    for sub,label in CATEGORY.items():
    # sub 要拼接的文件夹 label中文名
        for p in sorted((DOC_DIR / sub).glob("*.md")):
        # 拼接路径 排序 全局匹配 以md结尾
            docs.append({
                "doc_name": p.stem.split("_", 1)[-1],   # 01_财务报销管理制度 → 财务报销管理制度
                "category": label,
                "path": "docs/" + sub + "/" + p.name,   # 依然拼接路径
                "text": p.read_text(encoding="utf-8"),
            })
    return docs

if __name__ =="__main__":
    ds = load_docs() # 收集和组织结构化清单
    total = sum(len(d["text"]) for d in ds) # 一共多少字 [""]字典取值的标准写法
    print(f"读到{len(ds)}份文档，共{total}字")
    for d in ds[:3]:
        print("  -", d["doc_name"], "|", d["path"])
        # 只要前三分文件 名字加路径
        # 加载语料库 















