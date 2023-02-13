from .ExperimentBase import ExperimentBase
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import pandas as pd

class ModelEvaluation(ExperimentBase):
    def __init__(self, model, test_size=0.3, random_state=None):
        self.model = clone(model)
        self.y_pred = None
        self.class_labels = None
        self.y_test = None
        self.random_state = random_state
        self.test_size = test_size



    def run(self, X_train, y_train, X_test=pd.DataFrame([]), y_test=pd.DataFrame([])):
        if self.y_pred:
            raise Exception('Experiment already has results')

        if X_test.empty:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=self.test_size, random_state=self.random_state
            )

        try:
          self.model.fit(X_train, y_train)
          self.y_pred = self.model.predict(X_test)
          self.y_test = y_test
          self.class_labels = self.model.classes_.tolist()
          return True
        except Exception as e:
            print(e)
            return False


    def __repr__(self):
        return '<{0} model: {1}, class_labels: {2}>'.format(
            type(self).__name__,
            type(self.model).__name__,
            self.class_labels
        )