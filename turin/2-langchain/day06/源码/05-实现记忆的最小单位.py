



from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage

history = ChatMessageHistory()
history.add_user_message("hi!")
history.add_user_message("你好")
history.add_ai_message("whats up?")
# history.add_message(AIMessage(content="I don't know"))

print(history.messages)

