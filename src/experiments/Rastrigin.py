import mlrose_hiive as mlrose
import numpy as np

class Rastrigin:
    def __init__(self, state_size):
        self.state_size = state_size
        self.maximize = True
        self.normalization_metric = state_size
        self.problem = self.__build_problem(state_size)
# f(x) = An + sum(xi^2 - Acos(2pixi)), i=1 to n

    def __build_problem(self, state_size):
        def rastrigin_function(state):
            A = 10
            B = 0.1
            C = 0.0001*np.pi
            n = len(state)
            noise = np.random.uniform(-1.0, 0.0)
            # fitness = A*n + np.sum([(x**2 - A*np.cos(2*np.pi*x)) for x in state]) + noise
            fitness = A*n + np.sum([(x**2 - A*np.cos(2*np.pi*x) + B*np.sin(4*np.pi*x) + C*np.cos(6*np.pi*x)) for x in state])
            return fitness

        return mlrose.DiscreteOpt(
            length=state_size,
            fitness_fn=mlrose.CustomFitness(rastrigin_function),
            max_val=2,
            maximize=self.maximize
        )

# def modified_rastrigin(x, A=10, B=0.2, C=2*np.pi):
#     """
#     Computes the modified Rastrigin function for n inputs.
#     """
#     n = len(x)
#     f = A*n + np.sum(x**2 - A*np.cos(2*np.pi*x) + B*np.sin(4*np.pi*x) + C*np.cos(6*np.pi*x))
#     return f
