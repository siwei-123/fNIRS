import pandas as pd
import math


class CustomStandardScaler:
    def __init__(self):
        self.means = {}
        self.scales = {}

    def fit(self, x):
        variance = {}
        for column in x:
            self.means[column] = sum(x[column]) / len(x[column])
        for column in x:
            variance[column] = sum((x - self.means[column]) ** 2 for x in x[column]) / len(x[column])
            self.scales[column] = math.sqrt(variance[column])
        return self

    def transform(self, X):
        X_scaled = pd.DataFrame()
        for column in X.columns:
            X_scaled_list = []
            for x in X[column]:
                X_scaled_list.append((x - self.means[column]) / self.scales[column])
            X_scaled[column]=X_scaled_list
        return X_scaled

    def fit_transform(self, X):
        # 结合 fit 和 transform
        self.fit(X)
        return self.transform(X)

# 示例用法
