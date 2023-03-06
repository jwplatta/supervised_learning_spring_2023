import mlrose_hiive as mlrose
import numpy as np
import networkx as nx

class TravelingSalesman:
    def __init__(self, state_size=10, coords=[], distances=[]):
        self.coords = coords
        self.distances = distances
        self.maximize = False

        if coords:
            self.state_size = len(coords)
            fitness_fn = mlrose.TravellingSales(coords=coords)
        elif distances:
            self.state_size = len(distances)
            fitness_fn = mlrose.TravellingSales(distances=distances)
        else:
            self.state_size = state_size
            self.distances = self.__build_distances(state_size)
            fitness_fn = mlrose.TravellingSales(distances=self.distances)


        self.normalization_metric = sum([dist[2] for dist in self.distances])
        self.problem = self.__build_problem(self.state_size, fitness_fn)


    def __build_distances(self, city_count):
        graph = nx.gnp_random_graph(city_count, 1.0)
        distances = [(u, v, np.random.randint(1, 20)) for u, v in graph.edges()]
        # randomly assign a distance between 1 and 10 to each edge
        # coords = [(u,v) for u, v in graph.edges()]
        # graph[u][v]['weight'] = np.random.randint(1, 11)
        return distances #np.random.choice(np.arange(city_count**2), size=(city_count, 2), replace=False).tolist()


    def __build_problem(self, state_size, fitness_fn):
        return mlrose.TSPOpt(
            length=state_size,
            fitness_fn=fitness_fn,
            maximize=self.maximize
        )