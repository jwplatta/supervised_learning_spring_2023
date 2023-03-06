import mlrose_hiive as mlrose
import numpy as np
import networkx as nx

class FlipFlop:
    def __init__(self, state_size=10):
        self.maximize = True
        self.state_size = state_size
        self.normalization_metric = self.state_size
        self.problem = self.__build_problem(self.state_size)


    def __build_problem(self, state_size):
        return mlrose.DiscreteOpt(
            length=state_size,
            fitness_fn=mlrose.FlipFlop(),
            maximize=self.maximize
        )