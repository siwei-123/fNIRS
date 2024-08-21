class CustomLabelEncoder:
    def __init__(self):
        self.label_mapping = {}

    def fit(self, y):
        # 创建唯一标签到整数的映射
        unique_labels = sorted(set(y))  # 获取唯一标签，并排序以保持一致性
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, y):
        # 将标签转换为整数
        return [self.label_mapping[label] for label in y]

    def fit_transform(self, y):
        # 同时进行 fit 和 transform
        self.fit(y)
        return self.transform(y)


