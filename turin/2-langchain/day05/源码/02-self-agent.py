import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_openai import ChatOpenAI
load_dotenv()
# 将初始化工具，让它提供答案而不是文档
tools = [TavilyAnswer(name="Intermediate Answer", description="Answer Search")]
# 初始化大模型
llm = ChatOpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus"
)

# 获取使用提示 可以修改此提示    提示词模版
prompt = hub.pull("hwchase17/self-ask-with-search")

agent = create_self_ask_with_search_agent(llm, tools, prompt)
agent_exc = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
agent_exc.invoke({"input": "中国有哪些省份呢, 最大的省份是什么? 中文回答"})





