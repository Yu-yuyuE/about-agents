

from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# 指定要加载的Word文档路径
loader = UnstructuredWordDocumentLoader(r"人事管理流程.docx")
print(loader)

# 加载文档并分割成段落或元素
documents = loader.load()
print(documents)
# 输出加载的内容
for doc in documents:
    print(doc.page_content)

