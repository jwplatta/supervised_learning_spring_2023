import mlrose_hiive as mlrose
import numpy as np
import networkx as nx

class Knapsack:
    def __init__(self, state_size=None, weights=[], values=[], max_weight_pct=0.35, random_state=None):
        if random_state:
            np.random.seed(random_state)

        if state_size:
            self.state_size = state_size
            self.weights = np.random.randint(1, high=40, size=state_size)
            self.values = np.random.randint(1, high=20, size=state_size)
        else:
            self.state_size = len(weights)
            self.weights = weights
            self.values = values

        self.maximize = True
        self.normalization_metric = self.state_size
        self.problem = self.__build_problem(self.weights, self.values, max_weight_pct)


    def __build_problem(self, weights, values, max_weight_pct):
        return mlrose.DiscreteOpt(
            length=len(weights),
            fitness_fn=mlrose.Knapsack(weights, values, max_weight_pct),
            maximize=self.maximize
        )