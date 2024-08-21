import os
import pandas as pd

# 定义文件夹路径
folder_path = 'analyse_outcome'

# 初始化一个空的DataFrame，用于存储合并的数据
combined_df = pd.DataFrame()

# 遍历文件夹中的所有Excel文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):  # 检查是否为Excel文件
        file_path = os.path.join(folder_path, file_name)

        # 读取Excel文件，并将其数据追加到combined_df中
        df = pd.read_excel(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# 将合并后的数据保存到一个新的Excel文件
combined_df.to_excel('combined_file.xlsx', index=False)

print("所有Excel文件已合并并保存为 'combined_file.xlsx'")

