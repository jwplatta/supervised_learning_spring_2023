import sys
import os
sys.path.append("../")
import time

from src.datasets.SNLIData import SNLIData
from src.datasets.BankData import BankData

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import recall_score, make_scorer

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.SNLIData import SNLIData, SNLIFeaturesBERT

from src.experiments.CrossValidation import CrossValidation
from src.plots.CrossValidationFolds import CrossValidationFolds

from src.experiments.LearningCurve import LearningCurve
from src.plots.LearningCurve import LearningCurve as LearningCurvePlot

from src.experiments.ValidationCurve import ValidationCurve
from src.plots.ValidationCurve import ValidationCurve as ValidationCurvePlot

from src.experiments.ModelEvaluation import ModelEvaluation
from src.plots.ClassificationReport import ClassificationReport
from src.plots.ConfusionMatrix import ConfusionMatrix

random_state=42
bank_data = BankData(random_state=random_state, data_dir="./data")

#####################
### Decision Tree ###
#####################

untuned_dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced'
)

tuned_dt_clf = DecisionTreeClassifier(
    class_weight='balanced',
    criterion='gini',
    max_depth=9,
    min_samples_split=1100
)

untuned_learn_curv = LearningCurve(untuned_dt_clf, scoring='recall')
untuned_learn_curv.run(bank_data.X_train, bank_data.y_train)
untuned_learn_curv_plot = LearningCurvePlot(experiment=untuned_learn_curv)

tuned_learn_curv = LearningCurve(tuned_dt_clf, scoring='recall')
tuned_learn_curv.run(bank_data.X_train, bank_data.y_train)
tuned_learn_curv_plot = LearningCurvePlot(experiment=tuned_learn_curv)

tuned_cv = CrossValidation(tuned_dt_clf, scoring='recall')
tuned_cv.run(bank_data.X_train, bank_data.y_train)
tuned_cv_folds = CrossValidationFolds(experiment=tuned_cv)

dt_train_curv_fig, axs = plt.subplots(1, 3, figsize=(16, 4))
_, ax = untuned_learn_curv_plot.plot(ax=axs[0], title="max_depth=20, entropy", ymin=0.0, ymax=1.0)
_, ax = tuned_learn_curv_plot.plot(ax=axs[1], title="max_depth=9, min_samples_split=110, gini", ymin=0.0, ymax=1.0)
_, ax = tuned_cv_folds.plot(ax=axs[2])

dt_train_curv_fig.savefig(
    "./out/BankData/DecisionTree/LearningCurves - Report.png",
    bbox_inches='tight',
    dpi=800
)



