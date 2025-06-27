import pandas as pd
import os

# 原始数据路径
input_path = "yoochoose-data/yoochoose-clicks.dat"

# 正确加载：5列，分隔符为逗号
df = pd.read_csv(input_path, sep=",", header=None,
                 names=["SessionId", "Time", "ItemId", "Category", "Unknown"],
                 low_memory=False)

# 只保留 3 列
df = df[["SessionId", "ItemId", "Time"]]

# 将非法 ItemId 清除（确保为整数）
df = df[pd.to_numeric(df["ItemId"], errors="coerce").notnull()]
df["ItemId"] = df["ItemId"].astype("int64")

# 转换时间为时间戳（秒）
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time"])
df["Time"] = df["Time"].astype(int) // 10**9

# 排序
df = df.sort_values(["SessionId", "Time"])

# 拆分：最后 5000 个 session 为 test
sessions = df["SessionId"].drop_duplicates()
test_sess = sessions.tail(5000)
test_df = df[df["SessionId"].isin(test_sess)]
train_df = df[~df["SessionId"].isin(test_sess)]

# 创建输出目录
os.makedirs("data", exist_ok=True)

# 正确写入为 Tab 分隔 TSV
train_df.to_csv("data/train.tsv", sep="\t", index=False)
test_df.to_csv("data/test.tsv", sep="\t", index=False)

print("成功生成 data/train.tsv 和 data/test.tsv，完全符合 GRU4Rec 输入要求。")