import pandas as pd
import datasplitter
import customstandardscaler
import customclassweight
import classifier
from sklearn.metrics import confusion_matrix



file_path = 'combined_file.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')


df_cleaned = df.dropna()
df_cleaned=df_cleaned.drop(columns=['Channel'])

X = df_cleaned.drop(columns=['Quality'])
y = df_cleaned[['Quality']]

data_splitter =datasplitter.DataSplitter(df_cleaned, test_size=0.2, random_state=30)
x_train, x_test, y_train, y_test = data_splitter.split()


scaler = customstandardscaler.CustomStandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

scaler1 = customclassweight.CustomClassWeight()
scaler1.fit(y_train)


log_reg_model = classifier.LogisticRegressionManual(max_iter=2000, class_weight=scaler1.class_weight_dict)
log_reg_model.fit(X_train_scaled, y_train)


y_pred = log_reg_model.predict(X_test_scaled)




cm = confusion_matrix(y_test, y_pred)

B_accuracy=cm[0,0]/(cm[0,0]+cm[0,1])
G_accuracy=cm[1,1]/(cm[1,0]+cm[1,1])

Overall_accuracy=(cm[1,1]+cm[0,0])/(cm[1,1]+cm[0,0]+cm[1,0]+cm[0,1])

print('Bad accuracy:', B_accuracy)
print('Good accuracy:', G_accuracy)
print('Overall accuracy:', Overall_accuracy)