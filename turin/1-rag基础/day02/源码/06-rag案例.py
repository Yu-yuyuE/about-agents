from dotenv import load_dotenv
from openai import OpenAI
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import chromadb
import os


# 按照固定字符切割文档
def sliding_window_chunks(text, chunk_size, stride):
    return [text[i:i + chunk_size] for i in range(0, len(text), stride)]


def extract_text_from_pdf(pdf_filename, page_numbers=None):
    full_text = ''
    for i, page_layout in enumerate(extract_pages(pdf_filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for ele in page_layout:
            if isinstance(ele, LTTextContainer):
                # print(ele)
                # print(ele.get_text())
                full_text += ele.get_text().replace('\n', '').replace(' ', '')
    # print(full_text)
    # 按照固定字符切割文档
    chunks = sliding_window_chunks(full_text, chunk_size=250, stride=220)
    return chunks


# 向量数据库类
class MyVectorDBConnector:
    def __init__(self, collection_name):
        client = chromadb.PersistentClient(path=r"D:\python_project\vip_LLM\RAG\RAG-02\day02")
        # 创建一个 collection
        self.collection = client.get_or_create_collection(name=collection_name)

    # 使用智谱的模型进行向量化
    def get_embeddings(self, texts, model="text-embedding-v2"):
        '''封装 qwen 的 Embedding 模型接口'''
        # print('texts', texts)
        data = client.embeddings.create(input=texts, model=model).data
        return [x.embedding for x in data]

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.get_embeddings(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.get_embeddings([query]),
            n_results=top_n
        )
        return results


class RAG_Bot:
    def __init__(self, vector_db, n_results=5):
        self.vector_db = vector_db
        self.n_results = n_results

    def get_completion(self, prompt, model="qwen-plus"):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    # 聊天功能
    def chat(self, query):
        # 根据用户的问题在向量数据库进行检索
        results = self.vector_db.search(query, self.n_results)
        # 构建提示词
        prompt = prompt_template.replace('__INFO__', '\n'.join(results['documents'][0])).replace('__QUERY__', query)
        print('提示词:', prompt)
        response = self.get_completion(prompt)
        return response


if __name__ == '__main__':
    load_dotenv()
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_BASE_URL"))
    # __INFO__  需要根据用户的问题  在向量数据库当中  检索出相关的文档
    # __QUERY__   用户提的问题
    prompt_template = """
        你是一个问答机器人。
        你的任务是根据下述给定的已知信息回答用户问题。
        确保你的回复完全依据下述已知信息。不要编造答案。
        如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

        已知信息:
        鹅鹅鹅 曲项向天歌,白毛浮绿水,红掌拨清波。

        用户问：
        鹅鹅鹅的下一句是什么?
        请用中文回答用户问题。
        """

    # 使用示例
    docx_filename = "财务管理文档.pdf"
    # 读取Word文件
    # paragraphs = extract_text_from_docx(docx_filename, min_line_length=10)
    paragraphs = extract_text_from_pdf(docx_filename, page_numbers=[0, 1, 2])
    # 初始化向量数据库
    vector_db = MyVectorDBConnector('demo')
    # 添加文档到向量数据库
    vector_db.add_documents(paragraphs)

    bot = RAG_Bot(vector_db)
    response = bot.chat("财务管理权限划分?")
    print(response)
