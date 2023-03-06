import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
import mlrose_hiive as mlrose
import time


class NeuralNetwork(nn.Module):
    def __init__(self, n_layers, in_features, out_features, activation_fn):
        super(NeuralNetwork, self).__init__()
        self.in_features = in_features
        self.module_list = [
          nn.Linear(in_features, out_features),
          activation_fn()
        ]
        for layer_idx in range(n_layers-1):
            self.module_list.append(nn.Linear(out_features, out_features))
            self.module_list.append(activation_fn())

        self.network = nn.Sequential(*self.module_list)


    def forward(self, X):
        return self.network(X)


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
        n_layers=2, in_features=0,
        out_features=100,
        activation_fn=nn.ReLU,
        epochs=20,
        learning_rate=0.01,
        batch_size=1024,
        optimizer_strategy="backprop",
        max_attempts=10,
        max_iters=100,
        restarts=1,
        pop_size=200,
        mutation_prob=0.1,
        decay_type='geo',
        init_temp=1.0,
        decay=0.99,
        min_temp=0.001,
        exp_const=0.001,
        verbose=False
    ):
        self.n_layers = n_layers
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.classes_ = None
        self.model = None
        self.optimizer_strategy = optimizer_strategy

        # NOTE: params for mlrose optimizers
        self.max_iters = max_iters
        self.max_attempts = max_attempts
        self.restarts = restarts
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.decay_type=decay_type
        self.init_temp=init_temp
        self.decay=decay
        self.min_temp=min_temp
        self.exp_const=exp_const
        self.fitness_curves = []
        self.epoch_mean_func_evals = []
        self.total_func_evals = 0
        self.fit_time = None

        if self.optimizer_strategy == "backprop":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cpu"

        self.train_losses_ = []
        self.loss_ = None
        self.criterion = nn.CrossEntropyLoss()
        self.model = NeuralNetwork(self.n_layers, self.in_features, self.out_features, self.activation_fn)
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)
        self.optimizer.zero_grad()


    def fit(self, X, y):
        for param in self.model.parameters():
            if param.grad is not None:
                raise Exception('Model is trained. Use #partial_fit to continue training.')

        self.classes_ = np.unique(y)
        X_tensor = self.__to_tensor(X, torch.float)
        y_tensor = self.__to_tensor(y, torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.fitness_curves = []
        start_time = time.time()
        for epoch in range(self.epochs):
            batch_losses = []
            batch_func_evals = []

            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                if self.optimizer_strategy == "backprop":
                    loss = self.__backprop(X_batch, y_batch)
                    func_evals = 0
                elif self.optimizer_strategy == "random_hill_climb":
                    loss, fitness_curve = self.__random_hill_climbing(X_batch, y_batch)
                    func_evals = fitness_curve[-1, -1]
                    self.total_func_evals += func_evals
                elif self.optimizer_strategy == "simulated_annealing":
                    loss, fitness_curve = self.__simulated_annealing(X_batch, y_batch)
                    func_evals = fitness_curve[-1, -1]
                    self.total_func_evals += func_evals
                elif self.optimizer_strategy == "genetic_alg":
                    loss, fitness_curve = self.__genetic_alg(X_batch, y_batch)
                    func_evals = fitness_curve[-1, -1]
                    self.total_func_evals += func_evals

                batch_func_evals.append(func_evals)
                batch_losses.append(loss)

            self.epoch_mean_func_evals.append(np.mean(batch_func_evals))
            self.train_losses_.append(np.mean(batch_losses))

            if self.verbose:
                print('Epoch: {0} / {1} seconds - {2}'.format(epoch, time.time() - start_time, self.train_losses_[-1]))

        self.fit_time = time.time() - start_time

        return self


    def partial_fit(self, X, y):
        if type(y) == torch.Tensor:
            self.classes_ = torch.unique(y.cpu())
        else:
            self.classes_ = np.unique(y)

        X_tensor = self.__to_tensor(X, torch.float)
        y_tensor = self.__to_tensor(y, torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        batch_losses = []
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            if self.verbose:
                print('Batch: ', batch_idx)

            if self.optimizer_strategy == "backprop":
                loss = self.__backprop(X_batch, y_batch)
            elif self.optimizer_strategy == "random_hill_climb":
                loss, _ = self.__random_hill_climbing(X_batch, y_batch)
            elif self.optimizer_strategy == "simulated_annealing":
                loss, _ = self.__simulated_annealing(X_batch, y_batch)
            elif self.optimizer_strategy == "genetic_alg":
                loss, _ = self.__genetic_alg(X_batch, y_batch)

            batch_losses.append(loss)

        self.loss_ = np.mean(batch_losses)

        return self


    def predict(self, X, return_logits=False):
        with torch.no_grad():
            X_tensor = self.__to_tensor(X, torch.float)
            outputs = self.model(X_tensor)

            _, predicted = torch.max(outputs.data, 1)

            if return_logits:
                return predicted.cpu().numpy(), outputs
            else:
                return predicted.cpu().numpy()


    def __build_state_dict(self, weights_flat):
        param_shapes = [p.shape for p in self.model.parameters()]
        param_sizes = [p.flatten().shape[0] for p in self.model.parameters()]

        weights_split = torch.split(
            weights_flat, param_sizes
        )
        state_dict = {}
        for idx, key in enumerate(self.model.state_dict().keys()):
            state_dict[key] = torch.reshape(weights_split[idx], param_shapes[idx])

        return state_dict


    def __build_optimization_problem(self, X_tensor, y_tensor):
        def fitness_function(weights):
            weights_flat = torch.from_numpy(weights)
            # STEP: create the state dict
            state_dict = self.__build_state_dict(weights_flat)
            # STEP: load the weights
            self.model.load_state_dict(state_dict)
            # STEP: forward pass
            outputs = self.model(X_tensor)
            # STEP: calc loss
            loss = self.criterion(outputs, y_tensor)

            return loss.item()

        return mlrose.ContinuousOpt(
            length=torch.nn.utils.parameters_to_vector(self.model.parameters()).numel(),
            fitness_fn=mlrose.CustomFitness(fitness_function),
            step=self.learning_rate,
            maximize=False
        )


    def __backprop(self, X_batch, y_batch):
        outputs = self.model(X_batch)
        batch_loss = self.criterion(outputs, y_batch)
        loss = batch_loss.item()
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()
        return loss


    def __random_hill_climbing(self, X_batch, y_batch):
        init_state = torch.cat([p.detach().flatten() for p in self.model.parameters()]).numpy()

        updated_weights_flat, loss, fitness_curve = mlrose.random_hill_climb(
            self.__build_optimization_problem(X_batch, y_batch),
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            restarts=self.restarts,
            init_state=init_state,
            curve=True
        )
        state_dict = self.__build_state_dict(torch.from_numpy(updated_weights_flat))
        self.model.load_state_dict(state_dict)
        self.fitness_curves.append(fitness_curve)

        return loss, fitness_curve


    def __simulated_annealing(self, X_batch, y_batch):
        init_state = torch.cat([p.detach().flatten() for p in self.model.parameters()]).numpy()

        if self.decay_type == "geo":
            schedule = mlrose.GeomDecay(init_temp=self.init_temp, decay=self.decay, min_temp=self.min_temp)
        else:
            schedule = mlrose.ExpDecay(init_temp=self.init_temp, exp_const=self.exp_const, min_temp=self.min_temp),

        updated_weights_flat, loss, fitness_curve = mlrose.simulated_annealing(
            self.__build_optimization_problem(X_batch, y_batch),
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            schedule=schedule,
            init_state=init_state,
            curve=True
        )
        state_dict = self.__build_state_dict(torch.from_numpy(updated_weights_flat.copy()))
        self.model.load_state_dict(state_dict)
        self.fitness_curves.append(fitness_curve)

        return loss, fitness_curve


    def __genetic_alg(self, X_batch, y_batch):
        updated_weights_flat, loss, fitness_curve = mlrose.genetic_alg(
            self.__build_optimization_problem(X_batch, y_batch),
            max_attempts=self.max_attempts,
            max_iters=self.max_iters,
            pop_size=self.pop_size,
            mutation_prob=self.mutation_prob,
            curve=True
        )
        state_dict = self.__build_state_dict(torch.from_numpy(updated_weights_flat))
        self.model.load_state_dict(state_dict)

        self.fitness_curves.append(fitness_curve)

        return loss, fitness_curve


    def __to_tensor(self, data, dtype):
        data_type = type(data)

        if data_type != torch.Tensor:
            if data_type == pd.DataFrame or data_type == pd.Series:
                return torch.tensor(data.values, dtype=dtype).to(self.device)
            elif data_type == np.ndarray:
                return torch.tensor(data, dtype=dtype).to(self.device)
            else:
                raise Exception('{0} is none of tensor, dataframe, array.'.format(data_type))

        return data.to(self.device)