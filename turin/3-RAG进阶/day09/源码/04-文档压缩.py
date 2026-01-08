

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter, LLMChainFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()


# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# documents = TextLoader("deepseek介绍.txt", encoding="utf-8").load()
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1024,
#     chunk_overlap=100
# )
# texts = text_splitter.split_documents(documents)
# 本地embedding模型地址
embedding_model_path = r'D:\LLM\Local_model\BAAI\bge-large-zh-v1___5'
# 初始化嵌入模型（用于文本向量化）
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)
persist_directory="./chroma_compress"
# chroma = Chroma.from_documents(texts, embeddings_model, persist_directory=persist_directory)



chroma = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings_model
)
retriever = chroma.as_retriever()

print("-------------------压缩前--------------------------")
docs = retriever.invoke("deepseek的发展历程")
pretty_print_docs(docs)



llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

# # LLMChainExtractor 具体执行文档内容精炼的压缩器, 通过 LLM 对文档进行精炼
# compressor = LLMChainExtractor.from_llm(llm)
# # ContextualCompressionRetriever 将基础检索器和 压缩器结合
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )
# print("-------------------LLMChainExtractor压缩后--------------------------")
# compressed_docs = compression_retriever.invoke("deepseek的发展历程")
# pretty_print_docs(compressed_docs)



# LLMChainFilter 具体执行文档内容过滤的压缩器, 通过 LLM 对文档进行过滤
# _filter = LLMChainFilter.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=_filter, base_retriever=retriever
# )
# print("-------------------LLMChainFilter压缩后--------------------------")
# compressed_docs = compression_retriever.invoke("deepseek的发展历程")
# pretty_print_docs(compressed_docs)


# EmbeddingsFilter 具体执行文档内容过滤的压缩器, 通过检索到的文档相识度进行过滤  相似度 阈值  0.66 表示相识度大于 0.66 的文档才会被保留
# embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.74)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=embeddings_filter, base_retriever=retriever
# )
# compressed_docs = compression_retriever.invoke("deepseek的发展历程")
# print("-------------------EmbeddingsFilter压缩后--------------------------")
# pretty_print_docs(compressed_docs)


# EmbeddingsRedundantFilter是文档和文档之间进行过滤的压缩器
# 判断文档和文档直接的相似度    过滤的操作
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model)
# EmbeddingsFilter是查询问题和文档的相似度进行过滤
relevant_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.66)
# DocumentCompressorPipeline 是一个用于串联多个文档处理步骤的组件，其核心作用是通过组合不同的文档转换/过滤工具，构建一个多阶段的文档压缩流水线
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevant_filter]
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)
print("-------------------DocumentCompressorPipeline压缩后--------------------------")
compressed_docs = compression_retriever.invoke("deepseek的发展历程")
pretty_print_docs(compressed_docs)