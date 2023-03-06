import mlrose_hiive as mlrose
import numpy as np
import time
import copy


class FitnessValidationCurve:
    def __init__(self, model, problem, param_name, param_range, trials=1, state_size=100, name=None, verbose=False):
        self.model = model
        self.param_name = param_name
        self.param_range = param_range
        self.problem = problem
        self.state_size = state_size
        self.best_states = []
        self.best_fitnesses = []
        self.fitness_curves = []
        self.verbose = verbose
        self.trials = trials
        self.name = name


    def run(self):
        if self.verbose and self.name:
            print(self.name)

        for param_val in self.param_range:
            trial_fitnesses = []
            for trial in range(self.trials):
                model = copy.copy(self.model)
                setattr(model, self.param_name, param_val)

                problem = self.problem(state_size=self.state_size)

                best_state, best_fitness, fitness_curve = model.fit(
                  problem.problem,
                  curve=True
                )
                trial_fitnesses.append(best_fitness)

            best_fitness_mean = np.mean(trial_fitnesses)
            if self.verbose:
                print(model)
                print('{0}: {1} / best_fitness: {2}'.format(self.param_name, param_val, best_fitness_mean))

            self.best_states.append(best_state)
            self.best_fitnesses.append(best_fitness_mean)
            self.fitness_curves.append(fitness_curve)

        return True


    def __repr__(self):
        return '<{0} model: {1}>'.format(
            type(self).__name__,
            type(self.model).__name__
        )