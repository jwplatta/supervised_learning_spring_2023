import mlrose_hiive as mlrose


class GeneticAlgorithm:
    def __init__(self, population_size=200, mutation_prob=0.1, max_attempts=10, max_iters=100, curve=False, random_state=None):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.curve = curve
        self.random_state = random_state
        self.best_state = None
        self.best_fitness = None
        self.fitness_curve = None


    def fit(self, problem, curve=False):
        self.best_state, self.best_fitness, self.fitness_curve = mlrose.genetic_alg(
            problem,
            pop_size=self.population_size,
            mutation_prob=self.mutation_prob,
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            curve=curve,
            random_state=self.random_state
        )

        return [self.best_state, self.best_fitness, self.fitness_curve]


    def __repr__(self):
        return '<{0}, population_size: {1}, mutation_prob: {2}, max_attempts: {3}, max_iters: {4}>'.format(
            type(self).__name__,
            self.population_size,
            self.mutation_prob,
            self.max_attempts,
            self.max_iters
        )