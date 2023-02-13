import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from .constants import DATA_DIR


class BankData:
    SCALED_COLS=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    ONE_HOT_COLS=['job']
    ORDINAL_COLS=['month', 'marital', 'education', 'default', 'housing', 'loan', 'contact']
    TARGET_COL='y'

    def __init__(self, test_size=0.3, random_state=42, data_dir=None):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.filename = "bank-full.csv"
        self.test_size = test_size
        self.random_state = random_state
        self.classes = None
        self.class_labels = None
        self.class_labeler = LabelEncoder()
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = DATA_DIR

        self.__load()



    def __load(self):
        data = pd.read_csv(os.path.join(self.data_dir, self.filename), header=0, delimiter=";")
        self.X = data.drop(self.TARGET_COL, axis=1)
        self.y = data[self.TARGET_COL]
        y = self.class_labeler.fit_transform(self.y)
        self.classes = self.class_labeler.classes_.tolist()
        self.class_labels = self.class_labeler.transform(self.class_labeler.classes_).tolist()


        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=self.test_size, random_state=self.random_state)

        self.X_train_raw = X_train.copy()
        self.X_train_raw = self.X_train_raw.reset_index(drop=True)

        preprocessor = ColumnTransformer(
            transformers=[
                ('scaled', StandardScaler(), self.SCALED_COLS),
                ('one_hot', OneHotEncoder(), self.ONE_HOT_COLS),
                ('ordinal', OrdinalEncoder(), self.ORDINAL_COLS)
            ]
        )

        X_train = preprocessor.fit_transform(X_train)

        one_hot_cols = preprocessor.named_transformers_['one_hot'].get_feature_names_out()
        X_cols = self.SCALED_COLS + one_hot_cols.tolist() + self.ORDINAL_COLS

        self.X_train = pd.DataFrame(X_train, columns=X_cols)
        self.y_train = pd.Series(y_train)

        self.X_test = pd.DataFrame(preprocessor.transform(X_test), columns=X_cols)
        self.y_test = pd.Series(y_test)





