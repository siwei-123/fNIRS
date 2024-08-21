import math

# 示例数据
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [70000, 80000, 90000, 100000]
}

# 初始化空字典来存储每列的标准差
stds = {}

# 计算每列的标准差
for column in data:
    # 检查列的数据类型，忽略非数值类型的列
    if isinstance(data[column][0], (int, float)):
        # 计算均值
        mean = sum(data[column]) / len(data[column])
        # 计算方差
        variance = sum((x - mean) ** 2 for x in data[column]) / len(data[column])
        # 计算标准差
        stds[column] = math.sqrt(variance)

# 打印结果
print("Column standard deviations:", stds)
