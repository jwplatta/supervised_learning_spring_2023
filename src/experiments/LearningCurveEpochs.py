from .ExperimentBase import ExperimentBase
from sklearn.model_selection import KFold
from sklearn.base import clone
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import recall_score, accuracy_score


class LearningCurveEpochs(ExperimentBase):
    def __init__(self, model, batch_size=1024, scoring='cross-entropy', random_state=None, n_splits=5):
        self.model = clone(model)
        self.batch_size = batch_size
        self.scoring = scoring
        self.random_state = random_state
        self.n_splits = n_splits
        self.train_scores = []
        self.test_scores = []
        # self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = self.model.device


    def run(self, X, y):
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for epoch in range(self.model.epochs):
            fold_train_scores = []
            fold_test_scores = []

            for train_idx, test_idx in cv.split(X):
                X_train = X.iloc[train_idx]
                y_train = y[train_idx]

                X_test = X.iloc[test_idx]
                if self.scoring == "cross-entropy":
                    y_test = self.__to_tensor(y[test_idx], torch.long)
                else:
                    y_test = y[test_idx]

                self.model.partial_fit(X_train, y_train)

                if self.scoring == 'recall':
                    pred = self.model.predict(X_train)
                    score = recall_score(y_train, pred)
                elif self.scoring == 'accuracy':
                    pred = self.model.predict(X_train)
                    score = accuracy_score(y_train, pred)
                else:
                    score = self.model.loss_

                print('train: {}'.format(score))
                fold_train_scores.append(score)

                pred, logits = self.model.predict(X_test, return_logits=True)

                if self.scoring == 'recall':
                    score = recall_score(y_test, pred)
                elif self.scoring == 'accuracy':
                    score = accuracy_score(y_test, pred)
                else:
                    score = self.model.criterion(logits, y_test).item()

                print('test: {}'.format(score))
                fold_test_scores.append(score)

            self.train_scores.append(fold_train_scores)
            self.test_scores.append(fold_test_scores)

        self.train_scores = np.array(self.train_scores).mean(axis=1)
        self.test_scores = np.array(self.test_scores).mean(axis=1)


    def __to_tensor(self, data, dtype):
        data_type = type(data)

        if data_type != torch.Tensor:
            if data_type == pd.DataFrame or data_type == pd.Series:
                return torch.tensor(data.values, dtype=dtype).to(self.device)
            elif data_type == np.ndarray:
                return torch.tensor(data, dtype=dtype).to(self.device)
            else:
                raise Exception('{0} is none of tensor, dataframe, array.'.format(data_type))

        return data.to(self.device)


