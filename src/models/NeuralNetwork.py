import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class NeuralNetwork(nn.Module):
    def __init__(self, n_layers, in_features, out_features, activation_fn):
        super(NeuralNetwork, self).__init__()
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
    def __init__(self, n_layers=2, in_features=0, out_features=100, activation_fn=nn.ReLU, epochs=20, learning_rate=0.01, batch_size=1024, verbose=False):
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
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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

        for epoch in range(self.epochs):
            batch_losses = []

            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                outputs = self.model(X_tensor)

                batch_loss = self.criterion(outputs, y_tensor)
                batch_losses.append(batch_loss.item())

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            self.train_losses_.append(np.mean(batch_losses))
            if self.verbose:
                print('Epoch: ', epoch, ' - ', self.train_losses_[-1])

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

            outputs = self.model(X_batch)

            batch_loss = self.criterion(outputs, y_batch)
            batch_losses.append(batch_loss.item())

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

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