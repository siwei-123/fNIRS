import numpy as np
import pandas as pd


class LogisticRegressionManual:
    def __init__(self, max_iter=1000, class_weight=None, learning_rate=0.01):
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # 如果输入的是 DataFrame，转换为 numpy 数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            # 计算模型输出
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # 计算损失和梯度
            if self.class_weight is not None:
                sample_weights = np.array([self.class_weight[label] for label in y])
            else:
                sample_weights = np.ones(n_samples)

            # 交叉熵损失的梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y) * sample_weights)
            db = (1 / n_samples) * np.sum((y_pred - y) * sample_weights)

            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        # 如果输入的是 DataFrame，转换为 numpy 数组
        if isinstance(X, pd.DataFrame):
            X = X.values

        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        y_pred_prob = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_pred_prob]

