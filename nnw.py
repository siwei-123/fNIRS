import pandas as pd
import datasplitter
import customstandardscaler
import customclassweight
import keras
from keras import layers
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix




file_path = 'combined_file.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')


df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop(columns=['Channel'])


X = df_cleaned.drop(columns=['Quality'])
y = df_cleaned[['Quality']]


data_splitter = datasplitter.DataSplitter(df_cleaned, test_size=0.2, random_state=29)
x_train, x_test, y_train, y_test = data_splitter.split()





scaler = customstandardscaler.CustomStandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


scaler1 = customclassweight.CustomClassWeight()
scaler1.fit(y_train)


model = keras.Sequential(
    [
        layers.Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


y_train = y_train.values.ravel()
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, class_weight=scaler1.class_weight_dict)


y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)


cm = confusion_matrix(y_test, y_pred)
print(cm)


B_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])
G_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])
Overall_accuracy = (cm[1, 1] + cm[0, 0]) / (cm[1, 1] + cm[0, 0] + cm[1, 0] + cm[0, 1])



print('Bad accuracy:', B_accuracy)
print('Good accuracy:', G_accuracy)
print('Overall accuracy:', Overall_accuracy)

print(model.summary())