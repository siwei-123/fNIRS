import numpy as np


class CustomClassWeight:
    def __init__(self):
        self.class_weight_dict = {}
        self.unique_classes=2
        self.b_count=0
        self.g_count = 0

    def fit(self, Y):
        for value in Y['Quality']:
                if value==0:
                    self.b_count+=1
                elif value == 1:
                    self.g_count+=1

        total_samples = len(Y)


        self.class_weight_dict = {'1': total_samples / (self.unique_classes * self.g_count), '0': total_samples / (self.unique_classes * self.b_count)}
        return self





