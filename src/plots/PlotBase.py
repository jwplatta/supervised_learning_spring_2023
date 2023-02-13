import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os

class PlotBase:
    def __init__(self):
        pass


    def plot(self, ax=None, figsize=(8,5)):
        raise Exception("#plot not implemented")


    def save_obj(self, filename=None, path=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if filename:
            filename = "{0}.pickle".format(filename)
        else:
            filename = "{0}_{1}_{2}.pickle".format(type(self).__name__, self.model_name, timestamp)

        if path:
            full_path = os.path.join(path, filename)
        else:
            full_path = filename

        with open(full_path, "wb") as f:
            pickle.dump(self, f)


    def save_img(self, path=None):
        fig, ax = self.plot()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = "{0}_{1}_{2}.png".format(type(self).__name__, self.model_name, timestamp)

        if path:
            full_path = os.path.join(path, filename)
        else:
            full_path = filename

        try:
            fig.savefig(full_path)
            return True
        except Exception as e:
            print(e)
            return False


    def display(self):
        fig, ax = self.plot()
        plt.show()