import pandas as pd



# 原始数据
row = ["420374", "214537888", "2014-04-06T18:44:58.314Z"]

# 转换
session_id = int(row[0])
item_id = int(row[1])
timestamp = int(pd.to_datetime(row[2]).timestamp())

# 打印结果
print(">>> 开始测试")
print(f"{session_id}\t{item_id}\t{timestamp}")
print(repr(f"{session_id}\t{item_id}\t{timestamp}"))