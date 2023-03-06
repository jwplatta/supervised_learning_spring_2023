import mlrose_hiive as mlrose
import numpy as np


class RandomizedHillClimbing:
    def __init__(self, init_state=np.array([]), restarts=0, max_attempts=10, max_iters=np.inf, curve=False, random_state=None):
        self.restarts = restarts
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.curve = curve
        self.random_state = random_state
        self.best_state = None
        self.best_fitness = None
        self.fitness_curve = None
        self.init_state = init_state


    def fit(self, problem, curve=False):
        if not(self.init_state):
            self.init_state = self.__init_state(problem)

        self.best_state, self.best_fitness, self.fitness_curve = mlrose.random_hill_climb(
            problem,
            restarts=self.restarts,
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            init_state=self.init_state,
            curve=curve,
            random_state=self.random_state
        )

        return [self.best_state, self.best_fitness, self.fitness_curve]


    def __init_state(self, problem):
        if type(problem).__name__ == 'TSPOpt':
            return np.arange(problem.get_length())
        elif type(problem).__name__ == 'DiscreteOpt':
            return np.random.randint(0, problem.max_val, problem.get_length())
        else:
            raise Exception("unrecognized problem type")


    def __repr__(self):
        return '<{0}, restarts: {1}, max_attempts: {2}, max_iters: {3}>'.format(
            type(self).__name__,
            self.restarts,
            self.max_attempts,
            self.max_iters
        )