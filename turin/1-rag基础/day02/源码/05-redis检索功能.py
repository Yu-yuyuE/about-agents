import json
import redis

# 创建 Redis 连接
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)


# 从JSON文件中读取数据
def read_data():
    with open('train_zh.json', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # print(data[0:100])
    instructions = [entry['instruction'] for entry in data[0:1000]]
    outputs = [entry['output'] for entry in data[0:1000]]
    return instructions, outputs


# 将读取出来的数据存入Redis中
def set_redis_documents(instructions, outputs):
    for instruction, output in zip(instructions, outputs):
        r.set(instruction, output)


# 在Redis中根据关键词进行模糊搜索
def search_instructions(instruction_key, top_n=3):
    keys = r.keys(pattern='*' + instruction_key + '*')
    data = []
    for key in keys:
        data.append(r.get(key))
    return data[:top_n]


# 先从文件中读取数据
instructions, outputs = read_data()
# 在把数据存入到Redis中
set_redis_documents(instructions, outputs)
# 在Redis中进行检索
data = search_instructions('怀孕')
print(data)