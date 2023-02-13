import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import constants

class CovTypeData:
    SCALED_COLS=[
      'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
      'Horizontal_Distance_To_Fire_Points'
    ]
    OTHER_COLS=[
      'Wilderness_Area1', 'Wilderness_Area2',
      'Wilderness_Area3', 'Wilderness_Area4',
      'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
      'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
      'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
      'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
      'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
      'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
      'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
      'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
      'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
      'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
      'Soil_Type31', 'Soil_Type32', 'Soil_Type33',
      'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
      'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
      'Soil_Type40'
    ]
    TARGET_COL='Cover_Type'

    def __init__(self, test_size=0.3, random_state=42):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.filename = "covtype.csv"
        self.test_size = test_size
        self.random_state = random_state
        self.__load()


    def __load(self):
        data = pd.read_csv(os.path.join(constants.DATA_DIR, self.filename), header=0)
        X = data.drop(self.TARGET_COL, axis=1)
        y = data[self.TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        preprocessor = ColumnTransformer(
            transformers=[
                ('scaled', StandardScaler(), self.SCALED_COLS),
                ('', 'passthrough', self.OTHER_COLS)
            ]
        )

        self.X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=self.SCALED_COLS + self.OTHER_COLS)
        self.y_train = pd.Series(y_train)

        self.X_test = pd.DataFrame(preprocessor.transform(X_test), columns=self.SCALED_COLS + self.OTHER_COLS)
        self.y_test = pd.Series(y_test)