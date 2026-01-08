
import json
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


from langchain.schema import messages_from_dict, messages_to_dict
from dotenv import load_dotenv
import os

# 加载环境变量（需要包含API_KEY）
load_dotenv()


# 初始化大语言模型（通义千问）
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云兼容端点
    model="qwen-turbo"  # 使用qwen-turbo模型
)

# 创建对话提示模板
prompt = ChatPromptTemplate.from_messages([
    # 系统角色设定
    ("system", "你是一个友好的助手"),
    # 历史消息占位符（变量名必须与链配置中的history_messages_key一致）
    MessagesPlaceholder(variable_name="history"),
    # 用户输入占位符
    ("user", "{input}")
])

# 构建基础对话链（组合提示模板和语言模型）
base_chain = prompt | llm

store = {}
# {
#     第一个窗口对话:{ai:xxx,user:xxx, ai:xxx, user:xxx}
#     第二个窗口对话:{ai:xxx,user:xxx, ai:xxx, user:xxx}
#     第三个窗口对话:{ai:xxx,user:xxx, ai:xxx, user:xxx}
#  }
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def save_memory(filepath, session_id):
    """保存指定会话的历史记录到文件
    Args:
        filepath: 文件保存路径（建议使用.json扩展名）
        session_id: 要保存的会话ID（默认"default"）
    """
    history = get_session_history(session_id)
    # 将消息对象列表转换为字典格式
    dicts = messages_to_dict(history.messages)
    # 写入JSON文件（UTF-8编码）
    with open(filepath, "w", encoding='utf-8') as f:
        json.dump(dicts, f, ensure_ascii=False)


def load_memory(filepath, session_id):
    """从文件加载历史记录到指定会话
    Args:
        filepath: 历史记录文件路径
        session_id: 要加载到的会话ID（默认"default"）
    """
    with open(filepath, "r", encoding='utf-8') as f:
        dicts = json.load(f)
    # 将字典转换回消息对象列表
    messages = messages_from_dict(dicts)
    # 更新全局存储中的会话历史
    store[session_id] = ChatMessageHistory(messages=messages)



# 聊天记录对话链
conversation = RunnableWithMessageHistory(
    runnable=base_chain,   # 基础对话链
    get_session_history=get_session_history,  # 获取历史记录的方法
    input_messages_key='input',
    history_messages_key='history',
)

def legacy_predict(input_text, session_id):
    return conversation.invoke(
        {"input": input_text},  # 输入参数
        # 配置参数（必须包含session_id来关联历史记录）
        config={"configurable": {"session_id": session_id}}
    ).content



if __name__ == '__main__':

    session_id = 'aaaaa'
    # print(legacy_predict("你好", session_id))
    # print(legacy_predict("你是谁,我是柏汌", session_id))
    # save_memory("memory.json", session_id)

    load_memory("memory.json", session_id)
    print(legacy_predict("我是谁?", session_id))






