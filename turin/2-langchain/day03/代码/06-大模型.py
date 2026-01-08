from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
import os

load_dotenv()


# LLM纯文本补全模型
llm = ChatTongyi(api_key=os.getenv("api_key"),
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model='deepseek-v3')

text = "我的真的好想（帮我补全这个文本）"
res = llm.invoke(text)
print(res.content)