import pickle
import os
from datetime import datetime

class ExperimentBase:
    def __init__(self):
        pass


    def run(self):
        raise Exception('experiment not implemented')


    def save(self, path=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = "{0}_{1}_{2}.pickle".format(
            type(self).__name__,
            type(self.model).__name__,
            timestamp
        )

        if path:
            full_path = os.path.join(path, filename)
        else:
            full_path = filename

        with open(full_path, "wb") as f:
            pickle.dump(self, f)

