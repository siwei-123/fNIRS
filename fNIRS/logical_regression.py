import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import DataSplitter
import CustomStandardScaler
import CustomClassWeight
import classifier


file_path = 'combined_file.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')


df_cleaned = df.dropna()
df_cleaned=df_cleaned.drop(columns=['Channel'])

X = df_cleaned.drop(columns=['Quality'])
y = df_cleaned[['Quality']]



data_splitter = DataSplitter.DataSplitter(df_cleaned, test_size=0.2, random_state=30)
x_train, x_test, y_train, y_test = data_splitter.split()


scaler = CustomStandardScaler.CustomStandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)



scaler = CustomClassWeight.CustomClassWeight()
scaler.fit(y_train)


log_reg_model = classifier.LogisticRegressionManual(max_iter=1000, class_weight=scaler.class_weight_dict)
log_reg_model.fit(X_train_scaled, y_train)

# Step 5: 预测和评估
y_pred = log_reg_model.predict(X_test_scaled)


# 输出模型准确率和分类报告
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"模型准确率: {accuracy}")
print("分类报告:")
print(classification_rep)

