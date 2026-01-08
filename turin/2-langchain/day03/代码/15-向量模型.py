
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

embeddings_model = DashScopeEmbeddings(dashscope_api_key=os.getenv('api_key'))
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings), len(embeddings[0]), len(embeddings[1]))
##运行结果 (5, 1536)
