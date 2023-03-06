import mlrose_hiive as mlrose
import numpy as np

class SixPeaks:
    def __init__(self, state_size=10, threshold=0.1):
        self.maximize = True
        self.state_size = state_size
        self.normalization_metric = self.state_size
        self.problem = self.__build_problem(self.state_size, threshold)


    def __build_problem(self, state_size, threshold):
        return mlrose.DiscreteOpt(
            length=state_size,
            fitness_fn=mlrose.SixPeaks(t_pct=threshold),
            maximize=self.maximize
        )