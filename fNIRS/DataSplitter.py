import numpy as np

class DataSplitter:
    def __init__(self, data, test_size, random_state=None):
        self.data = data
        self.test_size = test_size
        self.random_state = random_state

    def split(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        count = int(len(self.data) * self.test_size)

        test_data = self.data.sample(n=count, replace=False)
        train_data = self.data.drop(test_data.index, axis=0)
        x_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        y_train = y_train.to_frame()
        x_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        y_test = y_test.to_frame()

        return x_train,x_test, y_train, y_test






