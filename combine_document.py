import os
import pandas as pd


folder_path = 'analyse_outcome_1'

combined_df = pd.DataFrame()


for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        file_path = os.path.join(folder_path, file_name)

        df = pd.read_excel(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)


combined_df.to_excel('combined_file.xlsx', index=False)



