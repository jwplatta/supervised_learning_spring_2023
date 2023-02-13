from .ExperimentBase import ExperimentBase
from sklearn.model_selection import cross_validate, KFold
from sklearn.base import clone
import numpy as np
import torch
from src.models.NeuralNetwork import NeuralNetworkClassifier

class CrossValidation(ExperimentBase):
    def __init__(self, model, scoring='accuracy', random_state=None, n_splits=5):
        self.model = clone(model)
        self.scoring = scoring
        self.random_state = random_state
        self.n_splits = n_splits
        self.fit_times = None
        self.score_times = None
        self.test_scores = None
        self.train_scores = None
        self.estimators = None
        self.result = None


    def run(self, X, y):
        if self.test_scores or self.train_scores:
            raise Exception('Experiment already has results')

        try:
            if type(self.model).__name__ == 'NeuralNetworkClassifier':
                n_jobs = 1
            else:
                n_jobs = 3

            result = cross_validate(
                self.model,
                X,
                y,
                cv=KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True),
                scoring=self.scoring,
                verbose=5,
                n_jobs=n_jobs,
                return_train_score=True,
                return_estimator=True
            )

            self.result = result
            self.fit_times = result['fit_time']
            self.score_times = result['score_time']
            self.test_scores = result['test_score']
            self.train_scores = result['train_score']
            self.estimators = result['estimator']

            return self
        except Exception as e:
            print(e)
            return False


    def __repr__(self):
        return '<{0} model: {1}, scoring: {2}, random_state: {3}, n_split: {4}>'.format(
            type(self).__name__,
            type(self.model).__name__,
            self.scoring,
            self.random_state,
            self.n_splits
        )
