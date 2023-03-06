import mlrose_hiive as mlrose
import numpy as np


class SimulatedAnnealing:
    def __init__(
      self,
      schedule=None,
      init_state=np.array([]),
      max_attempts=10,
      max_iters=100,
      decay_type='geo',
      init_temp=1.0,
      decay=0.99,
      min_temp=0.001,
      exp_const=0.001,
      random_state=None
    ):
        if schedule:
            self.schedule = schedule
        else:
            self.decay_type = decay_type
            self.decay = decay
            self.init_temp = init_temp
            self.min_temp = min_temp
            self.exp_const = exp_const
            self.schedule = None

        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.random_state = random_state
        self.best_state = None
        self.best_fitness = None
        self.fitness_curve = None
        self.init_state = init_state


    def fit(self, problem, curve=False):
        if not(self.init_state):
            self.init_state = self.__init_state(problem)

        self.best_state, self.best_fitness, self.fitness_curve = mlrose.simulated_annealing(
            problem,
            schedule=self.__build_schedule(),
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            init_state=self.init_state,
            curve=curve,
            random_state=self.random_state
        )

        return [self.best_state, self.best_fitness, self.fitness_curve]


    def __build_schedule(self):
        if self.schedule:
            return self.schedule
        elif self.decay_type == 'exp':
            return mlrose.ExpDecay(init_temp=self.init_temp, exp_const=self.exp_const, min_temp=self.min_temp)
        else:
            return mlrose.GeomDecay(init_temp=self.init_temp, decay=self.decay, min_temp=self.min_temp)


    def __init_state(self, problem):
        if type(problem).__name__ == 'TSPOpt':
            return np.arange(problem.get_length())
        elif type(problem).__name__ == 'DiscreteOpt':
            return np.random.randint(0, problem.max_val, problem.get_length())
        else:
            raise Exception("unrecognized problem type")


    def __repr__(self):
        return '<{0}, schedule: {1}, max_attempts: {2}, max_iters: {3}>'.format(
            type(self).__name__,
            self.schedule,
            self.max_attempts,
            self.max_iters
        )