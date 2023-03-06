from sklearn.base import clone
from sklearn.model_selection import train_test_split
from src.models.NeuralNetwork import NeuralNetworkClassifier
import numpy as np
import time

class CompareFitTimes:
    def __init__(self, model, name=None, random_state=None):
        self.name = name
        self.model = clone(model)
        self.fit_times = []
        self.train_sizes = None


    def run(self, X, y):
        train_sizes = np.linspace(0.1, 1.0, 5)
        sample_counts = []
        fit_times = []

        for train_size in train_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size-0.001)
            sample_counts.append(X_train.shape[0])
            model = clone(self.model)
            model.fit(X_train, y_train)
            fit_times.append(model.fit_time)

        self.train_sizes = np.array(sample_counts)
        self.fit_times = np.array(fit_times)

        return self.fit_times, self.train_sizes