from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化大模型:语言模型控制代理
llm = ChatOpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo"
)

# 设置工具:加载使用的工具，serpapi:调用Google搜索引擎
tools = load_tools(["serpapi"], llm=llm, SERPAPI_API_KEY=os.getenv("SERPAPI_API_KEY"))

# AgentType.ZERO_SHOT_REACT_DESCRIPTION   类型的代理   在没训练之前尝试解决问题
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
aa = agent.invoke({'input': "目前市场上苹果手机16的售价是多少？用中文回答"})
print(aa)





