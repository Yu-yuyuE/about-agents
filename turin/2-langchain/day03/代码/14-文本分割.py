



from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("财务管理文档.pdf")
pages = loader.load_and_split()
# print(f"第0页：\n{pages[0].page_content}")
# print(pages)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=100,
    length_function=len,
)

aa = text_splitter.create_documents([page.page_content.replace('\n', '').replace(' ', '') for page in pages if pages])
for i in aa:
    print(i.page_content, len(i.page_content))

