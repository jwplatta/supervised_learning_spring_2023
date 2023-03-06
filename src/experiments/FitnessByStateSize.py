import mlrose_hiive as mlrose
import numpy as np
import time
import copy


class FitnessByStateSize:
    def __init__(self, model, problem, state_sizes=np.arange(10, 120, 20), verbose=False):
        self.model = model
        self.problem = problem
        self.state_sizes = state_sizes
        self.best_states = []
        self.best_fitnesses = []
        self.best_fitnesses_normalized = []
        self.fitness_curves = []
        self.fit_times = []
        self.verbose = verbose


    def run(self):
        for state_size in self.state_sizes:
            model = copy.copy(self.model)
            problem = self.problem(state_size=state_size)

            start_time = time.time()
            best_state, best_fitness, fitness_curve = model.fit(
              problem.problem,
              curve=True
            )
            fit_time = time.time() - start_time

            if self.verbose:
                print(model)
                print(
                    'state_size: {0} / best_fitness: {1} / fit_time: {2}'.format(
                        state_size, best_fitness, fit_time
                    )
                )

            self.best_states.append(best_state)
            self.best_fitnesses.append(best_fitness)
            self.fitness_curves.append(fitness_curve)
            self.fit_times.append(fit_time)

        return True


    def __repr__(self):
        return '<{0} model: {1}>'.format(
            type(self).__name__,
            type(self.model).__name__
        )