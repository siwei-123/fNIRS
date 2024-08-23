import numpy as np

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
        X = X.values
        y = y.values

        n_samples= X.shape[0]
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)


            sample_weights_list=[]
            for i in y:
                if i==1:
                    sample_weights_list.append(self.class_weight["1"])
                elif i==0:
                    sample_weights_list.append(self.class_weight["0"])

            sample_weights=np.array(sample_weights_list)
            sample_weights=sample_weights.reshape(-1,1)

            y_pred = y_pred.reshape(-1, 1)
            cal_in_ad= y_pred - y
            cal_in_ad_use=cal_in_ad*sample_weights
            dw = (1 / n_samples) * np.dot(X.T, cal_in_ad_use )
            db = (1 / n_samples) * np.sum(cal_in_ad_use )

            self.weights=self.weights.reshape(-1,1)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db



    def predict_proba(self, X):

        X = X.values

        linear_model = np.dot(X, self.weights) + self.bias
        outcome=self.sigmoid(linear_model)
        return outcome

    def predict(self, X):
        y_pred_prob = self.predict_proba(X)
        result = []
        for i in y_pred_prob:
            if i > 0.5:
                result.append(1)
            else:
                result.append(0)
        return result




