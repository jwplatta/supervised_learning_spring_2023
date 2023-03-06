import mlrose_hiive as mlrose
import numpy as np
import networkx as nx

class MaxKColor:
    def __init__(self, edges=[], state_size=None):
        self.maximize = True
        if state_size:
            self.state_size = state_size
            graph = nx.gnp_random_graph(state_size, 1.0)
            self.edges = list(graph.edges())
        else:
            self.state_size = len(edges)
            self.edges = edges

        self.normalization_metric = len(self.edges)
        self.problem = self.__build_problem(self.edges)


    def __build_problem(self, edges):
        return mlrose.DiscreteOpt(
            length=len(edges),
            fitness_fn=mlrose.MaxKColor(edges),
            maximize=self.maximize
        )