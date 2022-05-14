import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, additional_param="log"):
        self.additional_param = additional_param

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if self.additional_param == "log":
            # логарифмирование
            eps = 1e-6
            X_ = np.log(X_ + eps)
        elif self.additional_param == "square":
            # возведение в квадрат
            X_ = X_ ** 2
        else:
            raise NotImplementedError
        return X_
