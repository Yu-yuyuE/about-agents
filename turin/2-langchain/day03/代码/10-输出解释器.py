from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# 创建解析器
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, XMLOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化语言模型
model = ChatOpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# output_parser = StrOutputParser()
output_parser = JsonOutputParser()
# xml_parser = XMLOutputParser()

# 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的程序员"),
    ("user", "{input}")
])

# 将提示和模型合并以进行调用
# chain = prompt | model | output_parser
chain = prompt | model | output_parser

# res = chain.invoke({"input": "langchain是什么? 使用xml格式输出"})
# 输出的数据的类型是输出解释器提供的
res = chain.invoke({"input": "langchain是什么? "})
# res = chain.invoke({"input": "大模型中的langchain是什么?"})
print(res)