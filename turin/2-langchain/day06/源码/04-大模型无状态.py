

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model_name="qwen-plus")

# 直接提供问题，并调用llm
response = llm.invoke("你好我是柏汌")
# print(response)
# print("=" * 50)
print(response.content)

response = llm.invoke("我是谁?")

print(response.content)


