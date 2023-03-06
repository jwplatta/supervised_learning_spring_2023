import mlrose_hiive as mlrose
import numpy as np
import time
import copy


class FitnessLearningCurve:
    def __init__(self, model, problem, state_size=100, max_iters=np.arange(1, 125, 25), verbose=False):
        self.model = model
        self.problem = problem
        self.max_iters = max_iters
        self.state_size = state_size
        self.best_states = []
        self.best_fitnesses = []
        self.fitness_curves = []
        self.verbose = verbose


    def run(self):
        for max_iters in self.max_iters:
            model = copy.copy(self.model)
            model.max_attempts = 2
            model.max_iters = max_iters

            best_state, best_fitness, fitness_curve = model.fit(
              self.problem(self.state_size).problem,
              curve=True
            )

            if self.verbose:
                print(model)
                print('max_iters: {0} / best_state: {1} / best_fitness: {2}'.format(max_iters, best_state, best_fitness))

            self.best_states.append(best_state)
            self.best_fitnesses.append(best_fitness)
            self.fitness_curves.append(fitness_curve)

        return True


    def __repr__(self):
        return '<{0} model: {1}>'.format(
            type(self).__name__,
            type(self.model).__name__
        )