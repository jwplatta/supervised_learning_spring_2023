from .ExperimentBase import ExperimentBase
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class BoostingIterations(ExperimentBase):
    def __init__(self, model, scoring='accuracy', random_state=None, n_splits=3):
        self.model = model
        self.scoring = scoring
        self.random_state = random_state
        self.n_splits = n_splits
        self.train_losses = []
        self.test_losses = []
        self.estimators = []


    def run(self, X, y):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        train_scores = []
        test_scores = []

        for train_idx, test_idx in cv.split(X):
            model_clone = clone(self.model)
            self.estimators.append(model_clone)
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]

            X_test = X.iloc[test_idx]
            y_test = y[test_idx]

            model_clone.fit(X_train, y_train)

            staged_train_scores = []
            for i, y_pred in enumerate(model_clone.staged_predict(X_train)):
                score = self.__score(y_train, y_pred)
                staged_train_scores.append(score)

            train_scores.append(staged_train_scores)

            staged_test_scores = []
            for i, y_pred in enumerate(model_clone.staged_predict(X_test)):
                score = self.__score(y_test, y_pred)
                staged_test_scores.append(score)

            test_scores.append(staged_test_scores)

        self.train_scores = np.array(train_scores).mean(axis=0)
        self.test_scores = np.array(test_scores).mean(axis=0)


    def __score(self, y_test, y_pred):
        if self.scoring == 'balanced_accuracy':
            return balanced_accuracy_score(y_test, y_pred)
        elif self.scoring == 'accuracy':
            return accuracy_score(y_test, y_pred)
        elif self.scoring == 'f1':
            return f1_score(y_test, y_pred)
        elif self.scoring == 'recall':
            return recall_score(y_test, y_pred)
        elif self.scoring == 'precision':
            return precision_score(y_test, y_pred)
        else:
            raise Exception('Unrecognized scoring metric')