


import chromadb

#创建链接
# client = chromadb.Client()
# 持久化存储    默认是存在内存   程序结束就会丢失
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name='test')

# 增加
collection.add(
    documents=["Article by john", "Article by Jack", "Article by Jill"], # 文本内容列表，每个元素是一段文本（如文章、句子等）
    embeddings=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 嵌入向量列表，每个元素是一个与 documents 对应的向量表示
    ids=["1", "2", "3"] # 自定义 ID 列表，用于唯一标识每条记录

)

# 查询
aa = collection.get(
    # ids=['1'],
    # include=['documents', 'embeddings']
    where_document={'$contains': 'Jack'}
)
# print(aa)


# 删除
collection.delete(
    ids=['2']
)
print(collection.get())


# 修改
collection.update(
    documents=["Article by john", "Article by Jack", "Article by Jill"],
    embeddings=[[10,2,3],[40,5,6],[70,8,9]],
    ids=["1", "2", "3"])
print(collection.get(include=["embeddings"]))











