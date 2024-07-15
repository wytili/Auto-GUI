import json
import random

filename = '/data/zzs800-0/wangyt/seeclick_web.json'

with open(filename, 'r') as file:
    data = json.load(file)

sample_size = len(data) // 10

sampled_data = random.sample(data, sample_size)

# 将采样结果保存到新的JSON文件
with open('/data/zzs800-0/wangyt/seeclick_web_27k.json', 'w') as file:
    json.dump(sampled_data, file, indent=4)