
from langchain_openai import  ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

# 创建模型实例
model = ChatOpenAI(api_key=os.getenv("api_key"),
                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   model='qwen-plus')

Prompt = PromptTemplate(
    template='你是一个专业的程序员, 对{test}进行表述'
)

input = Prompt.format(test='python')
# print(input)
resource = model.invoke(input)
print(resource)











