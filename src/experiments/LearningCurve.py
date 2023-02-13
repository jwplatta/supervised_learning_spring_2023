from .ExperimentBase import ExperimentBase
from sklearn.model_selection import learning_curve, KFold
import numpy as np
from sklearn.base import clone

class LearningCurve(ExperimentBase):
    def __init__(self, model, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), random_state=None, n_splits=5):
        self.model = clone(model)
        self.scoring = scoring
        self.train_sizes = train_sizes
        self.n_splits = n_splits
        self.train_scores = None
        self.test_scores = None
        self.fit_times = None
        self.score_times = None


    def run(self, X, y):
        if self.train_scores != None or self.test_scores != None:
            raise Exception('Experiment already has results')

        try:
            if type(self.model).__name__ == 'NeuralNetworkClassifier':
                n_jobs = 1
            else:
                n_jobs = 3

            train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
                self.model,
                X,
                y,
                cv=KFold(n_splits=self.n_splits, shuffle=True),
                scoring=self.scoring,
                verbose=5,
                n_jobs=n_jobs,
                shuffle=True,
                return_times=True
            )

            self.train_scores = train_scores.mean(axis=1)
            self.test_scores = test_scores.mean(axis=1)
            self.fit_times = fit_times.mean(axis=1)
            self.score_times = score_times.mean(axis=1)
            self.train_sizes = train_sizes

            return True
        except Exception as e:
            print(e)
            return False


        def __repr__(self):
            return '<{0} model: {1}, scoring: {2}, train_sizes: {3}, random_state: {4}, n_split: {5}>'.format(
                type(self).__name__,
                type(self.model).__name__,
                self.scoring,
                self.train_sizes,
                self.random_state,
                self.n_splits
            )


