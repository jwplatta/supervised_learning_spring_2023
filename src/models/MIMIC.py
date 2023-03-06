import mlrose_hiive as mlrose

class MIMIC:
    def __init__(self, population_size=100, keep_pct=0.2, max_attempts=10, max_iters=100, random_state=None):
        self.population_size = population_size
        self.keep_pct = keep_pct
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.random_state = random_state
        self.best_state = None
        self.best_fitness = None
        self.fitness_curve = None


    def fit(self, problem, curve=False):
        self.best_state, self.best_fitness, self.fitness_curve = mlrose.mimic(
            problem,
            pop_size=self.population_size,
            keep_pct=self.keep_pct,
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            curve=curve,
            random_state=self.random_state
        )

        return [self.best_state, self.best_fitness, self.fitness_curve]


    def __repr__(self):
        return '<{0}, population_size: {1}, keep_pct: {2}, max_attempts: {3}, max_iters: {4}>'.format(
            type(self).__name__,
            self.population_size,
            self.keep_pct,
            self.max_attempts,
            self.max_iters
        )
