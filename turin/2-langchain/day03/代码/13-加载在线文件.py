
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
data = loader.load()
print(f"第0页：\n{data[0].page_content}")  # 也可通过 pages[0].page_content只获取本页内容
# 需要注意科学上网
