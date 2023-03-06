import time
import numpy as np
import copy

class EvalOptimizationAlgorithm:
    def __init__(self, opt_algorithm, problem, trials=3, verbose=False):
        self.opt_algorithm = opt_algorithm
        self.problem = problem
        self.trials = trials
        self.verbose = verbose
        self.fitness_curves = []

    def fit(self):
        trial_fitnesses = []
        trial_fit_times = []
        trial_iters = []

        for trial in range(1, self.trials+1):

            opt_algorithm = copy.copy(self.opt_algorithm)
            problem = copy.copy(self.problem)

            start_time = time.time()
            best_state, best_fitness, fitness_curve = opt_algorithm.fit(
              problem.problem,
              curve=True
            )
            fit_time = time.time() - start_time

            if self.verbose:
                print('{0} - {1}:\n\tbest_fitness {2}\n\tfit_time {3}\n\tfunc_calls {4}\n\titerations {5}'.format(
                  trial, opt_algorithm, best_fitness, fit_time, fitness_curve[-1, -1], fitness_curve.shape[0]
                ))

            trial_fitnesses.append(best_fitness)
            trial_fit_times.append(fit_time)
            self.fitness_curves.append(fitness_curve)
            trial_iters.append(fitness_curve.shape[0])

        self.fit_times = np.array(trial_fit_times)
        self.mean_fit_time = np.mean(trial_fit_times)

        self.fitnesses = np.array(trial_fitnesses)
        self.mean_fitness = np.mean(trial_fitnesses)

        self.eval_func_calls = np.array([curv[-1, -1] for curv in self.fitness_curves])
        self.mean_eval_func_calls = np.mean(self.eval_func_calls)

        self.iterations = np.array(trial_iters)
        self.mean_iterations = np.mean(trial_iters)

        if self.verbose:
            print(
              '{0}:\n\tMean fitness: {1}\n\tMean Fit Time: {2}\n\tMean Eval Func Calls: {3}\n\tMean Iterations: {4}'.format(
                self.opt_algorithm, self.mean_fitness, self.mean_fit_time, self.mean_eval_func_calls, self.mean_iterations
              )
            )

        return self.mean_fitness, self.mean_fit_time





