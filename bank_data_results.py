import sys
import os
# sys.path.append("../")
import time

from src.datasets.SNLIData import SNLIData
from src.datasets.BankData import BankData

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import recall_score, make_scorer

import torch
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src.models.NeuralNetwork import NeuralNetworkClassifier

from src.plots.CrossValidationFolds import CrossValidationFolds
from src.plots.ValidationCurve import ValidationCurve as ValidationCurvePlot
from src.plots.Histogram import Histogram
from src.plots.LearningCurve import LearningCurve as LearningCurvePlot
from src.plots.ClassificationReport import ClassificationReport
from src.plots.ConfusionMatrix import ConfusionMatrix
from src.plots.ScatterPlot import ScatterPlot
from src.plots.CompareFitScoreTimes import CompareFitScoreTimes
from src.plots.BoostingIterations import BoostingIterations as BoostingIterationsPlot
from src.plots.LearningEpochsCurve import LearningEpochsCurve as LearningEpochsCurvePlot

from src.experiments.CrossValidation import CrossValidation
from src.experiments.ValidationCurve import ValidationCurve
from src.experiments.ModelEvaluation import ModelEvaluation
from src.experiments.LearningCurve import LearningCurve
from src.experiments.BoostingIterations import BoostingIterations
from src.experiments.LearningCurveEpochs import LearningCurveEpochs

working_dir = os.getcwd()
out_dir = os.path.join(working_dir, "out")
data_file = os.path.join(working_dir, "data", "bank-full.csv")


if not os.path.exists(out_dir):
    os.mkdir(out_dir)


if not os.path.exists(data_file):
    raise Exception("Please get the data from the dropbox link in the readme.txt")

plt.style.use('seaborn')

random_state=42
bank_data = BankData(
    data_dir=os.path.join(working_dir, "data"),
    random_state=random_state
)

########################
### Data Exploration ###
########################
class_imbalance_plot = Histogram(
    bank_data.y_train, labels=bank_data.y_train,
    xticks=bank_data.classes,
    title="Target Class Imbalance"
)

instances = bank_data.X_train_raw.copy()
instances['target'] = bank_data.y_train.copy()

pdays_scatter_plot = ScatterPlot(
    instances, 'duration', 'pdays', 'target',
    label_names=bank_data.classes
)

contact_scatter_plot = ScatterPlot(
    instances,
    'duration', 'previous', 'target',
    label_names=bank_data.classes
)

instances = bank_data.X_train.copy()
instances['target'] = bank_data.y_train.copy()
corr_matrix = instances[['duration', 'pdays', 'previous', 'housing', 'contact', 'target']].corr()


data_exp_fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
_, ax = class_imbalance_plot.plot(ax=axs[0])
# sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, ax=axs[1])
_, ax = pdays_scatter_plot.plot(ax=axs[1], title="Scatter A")
_, ax = contact_scatter_plot.plot(ax=axs[2], title="Scatter B")


data_exp_fig.savefig(
    os.path.join(out_dir, "Data Exploration - Report - C.png"),
    bbox_inches='tight',
    dpi=800
)


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

dt_untuned_learning_curve = LearningCurve(untuned_dt_clf, scoring='recall')
dt_untuned_learning_curve.run(bank_data.X_train, bank_data.y_train)
dt_untuned_learning_curve_plot = LearningCurvePlot(experiment=dt_untuned_learning_curve)

dt_tuned_learning_curve = LearningCurve(tuned_dt_clf, scoring='recall')
dt_tuned_learning_curve.run(bank_data.X_train, bank_data.y_train)
dt_tuned_learning_curve_plot = LearningCurvePlot(experiment=dt_tuned_learning_curve)

dt_tuned_cv = CrossValidation(tuned_dt_clf, scoring='recall')
dt_tuned_cv.run(bank_data.X_train, bank_data.y_train)
dt_tuned_cv_folds = CrossValidationFolds(experiment=dt_tuned_cv)

dt_train_curv_fig, axs = plt.subplots(1, 3, figsize=(20, 5))
_, ax = dt_untuned_learning_curve_plot.plot(ax=axs[0], ymin=0.3, ymax=1.01)
_, ax = dt_tuned_learning_curve_plot.plot(ax=axs[1], ymin=0.3, ymax=1.01)
_, ax = dt_tuned_cv_folds.plot(ax=axs[2])

dt_train_curv_fig.savefig(
    os.path.join(out_dir, "BankData", "DecisionTree", "LearningCurves - Report - C.png"),
    bbox_inches='tight',
    dpi=800
)

val_curv = ValidationCurve(
    DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'),
    'min_samples_leaf',
    [
        1, 10, 100, 1000
    ],
    scoring='recall',
)
val_curv.run(bank_data.X_train, bank_data.y_train)
min_samples_leaf_plot = ValidationCurvePlot(experiment=val_curv)

val_curv = ValidationCurve(
    DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'),
    'max_depth',
    [
        2, 4, 6, 8, 10, 12
    ],
    scoring='recall',
)
val_curv.run(bank_data.X_train, bank_data.y_train)
max_depth_plot = ValidationCurvePlot(experiment=val_curv)

val_curv = ValidationCurve(
    DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'),
    'min_samples_split',
    [
        2, 10, 100, 1000, 1500
    ],
    scoring='recall',
)
val_curv.run(bank_data.X_train, bank_data.y_train)
min_samples_split_plot = ValidationCurvePlot(experiment=val_curv)

validation_fig, axs = plt.subplots(1, 3, figsize=(16, 4))
_, ax = max_depth_plot.plot(ax=axs[0])
_, ax = min_samples_split_plot.plot(ax=axs[1])
_, ax = min_samples_leaf_plot.plot(ax=axs[2])

validation_fig.savefig(
    os.path.join(out_dir, "BankData", "DecisionTree", "ValidationCurves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

tuned_model_eval = ModelEvaluation(tuned_dt_clf)
tuned_model_eval.run(bank_data.X_train, bank_data.y_train, bank_data.X_test, bank_data.y_test)
print(tuned_model_eval.model)
print(ClassificationReport(experiment=tuned_model_eval).plot())


##############################
### Support Vector Machine ###
##############################

untuned_svm_clf = LinearSVC(
    C=1.0, max_iter=1000, penalty='l2', dual=False, verbose=0, class_weight='balanced'
)
# SVC(C=0.01, class_weight='balanced', gamma=0.15, verbose=0)
# tuned_svm_clf = SVC(C=0.15, gamma=0.2, kernel='rbf', class_weight='balanced', verbose=0)
tuned_svm_clf = SVC(C=0.1, gamma=0.12, kernel='rbf', class_weight='balanced', verbose=0)

svm_untuned_learning_curve = LearningCurve(
    untuned_svm_clf,
    scoring='recall',
    n_splits=3
)
svm_untuned_learning_curve.run(bank_data.X_train, bank_data.y_train)

svm_tuned_learning_curve = LearningCurve(
    tuned_svm_clf,
    scoring='recall',
    n_splits=3
)
svm_tuned_learning_curve.run(bank_data.X_train, bank_data.y_train)

svm_tuned_cv = CrossValidation(tuned_svm_clf, scoring='recall')
svm_tuned_cv.run(bank_data.X_train, bank_data.y_train)
svm_tuned_cv_folds = CrossValidationFolds(experiment=svm_tuned_cv)

svm_untuned_learning_curve_plot = LearningCurvePlot(experiment=svm_untuned_learning_curve)
svm_tuned_learning_curve_plot = LearningCurvePlot(experiment=svm_tuned_learning_curve)

svm_learning_curves_fig, axs = plt.subplots(1, 3, figsize=(18, 5))
_, ax = svm_untuned_learning_curve_plot.plot(ax=axs[0], ymin=0.5, ymax=1.01)
_, ax = svm_tuned_learning_curve_plot.plot(ax=axs[1], ymin=0.5, ymax=1.01)
_, ax = svm_tuned_cv_folds.plot(ax=axs[2])

svm_learning_curves_fig.savefig(
    os.path.join(out_dir, "BankData", "SVM", "LearningCurves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

svc_C_val_curv = ValidationCurve(
    SVC(gamma=0.1, kernel='rbf', class_weight='balanced', verbose=0),
    'C',
    [0.001, 0.01, 0.1, 1.0, 10],
    scoring='recall',
    n_splits=3
)
svc_C_val_curv.run(bank_data.X_train, bank_data.y_train)

svc_gamma_val_curv = ValidationCurve(
    SVC(C=0.1, kernel='rbf', class_weight='balanced', verbose=0),
    'gamma',
    [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0],
    scoring='recall',
    n_splits=3
)
svc_gamma_val_curv.run(bank_data.X_train, bank_data.y_train)

svc_C_val_curv_plot = ValidationCurvePlot(experiment=svc_C_val_curv)
svc_gamma_val_curv_plot = ValidationCurvePlot(experiment=svc_gamma_val_curv)

svc_val_curvs_fig, axs = plt.subplots(1, 2, figsize=(10, 4))
_, ax = svc_C_val_curv_plot.plot(ax=axs[0])
_, ax = svc_gamma_val_curv_plot.plot(ax=axs[1])

svc_val_curvs_fig.savefig(
    os.path.join(out_dir, "BankData", "SVM", "Validation Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

tuned_model_eval = ModelEvaluation(tuned_svm_clf)
tuned_model_eval.run(bank_data.X_train, bank_data.y_train, bank_data.X_test, bank_data.y_test)
print(tuned_model_eval.model)
print(ClassificationReport(experiment=tuned_model_eval).plot())

##########################
### k-Nearest Neighbor ###
##########################
knn_untuned_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
knn_tuned_clf = KNeighborsClassifier(n_neighbors=6, weights='uniform', p=2)

knn_tuned_cv = CrossValidation(knn_tuned_clf, scoring='recall')
knn_tuned_cv.run(bank_data.X_train, bank_data.y_train)
knn_tuned_cv_folds = CrossValidationFolds(experiment=knn_tuned_cv)

knn_untuned_learning_curve = LearningCurve(knn_untuned_clf, scoring='recall')
knn_untuned_learning_curve.run(bank_data.X_train, bank_data.y_train)
knn_untuned_learning_curve_plot = LearningCurvePlot(experiment=knn_untuned_learning_curve)

knn_tuned_learning_curve = LearningCurve(knn_tuned_clf, scoring='recall')
knn_tuned_learning_curve.run(bank_data.X_train, bank_data.y_train)
knn_tuned_learning_curve_plot = LearningCurvePlot(experiment=knn_tuned_learning_curve)

knn_train_curv_fig, axs = plt.subplots(1, 3, figsize=(16, 4))
_, ax = knn_untuned_learning_curve_plot.plot(ax=axs[0], ymin=0.0, ymax=1.02)
_, ax = knn_tuned_learning_curve_plot.plot(ax=axs[1], ymin=0.0, ymax=1.02)
_, ax = knn_tuned_cv_folds.plot(ax=axs[2])

knn_train_curv_fig.savefig(
    os.path.join(out_dir, "BankData", "KNN", "Training Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

uniform_val_curv = ValidationCurve(
    KNeighborsClassifier(weights='uniform', p=2),
    'n_neighbors',
    np.arange(2, 45, 5),
    scoring='recall'
)
uniform_val_curv.run(bank_data.X_train, bank_data.y_train)

distance_val_curv = ValidationCurve(
    KNeighborsClassifier(weights='distance', p=2, algorithm='ball_tree', leaf_size=1000),
    'n_neighbors',
    np.arange(1, 13, 2),
    scoring='recall',
    n_splits=3
)
distance_val_curv.run(bank_data.X_train, bank_data.y_train)

uniform_val_curv_plot = ValidationCurvePlot(experiment=uniform_val_curv)
distance_val_curv_plot = ValidationCurvePlot(experiment=distance_val_curv)

knn_val_curvs_fig, axs = plt.subplots(1, 2, figsize=(10, 4))
_, ax = uniform_val_curv_plot.plot(ax=axs[0], title="weights=uniform")
_, ax = distance_val_curv_plot.plot(ax=axs[1], title="weights=distance")

knn_val_curvs_fig.savefig(
    os.path.join(out_dir, "BankData", "KNN", "Validation Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

knn_tuned_model_eval = ModelEvaluation(knn_tuned_clf)
knn_tuned_model_eval.run(bank_data.X_train, bank_data.y_train, bank_data.X_test, bank_data.y_test)
print(knn_tuned_model_eval.model)
print(ClassificationReport(experiment=knn_tuned_model_eval).plot())

################
### AdaBoost ###
################
dstump = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
ada_untuned_clf = AdaBoostClassifier(base_estimator=dstump, n_estimators=300, learning_rate=1.0)
tuned_weak = DecisionTreeClassifier(class_weight='balanced', min_samples_split=8000)
ada_tuned_clf = AdaBoostClassifier(base_estimator=tuned_weak, n_estimators=500, learning_rate=0.01)

ada_untuned_learning_curve = LearningCurve(ada_untuned_clf, scoring='recall')
ada_untuned_learning_curve.run(bank_data.X_train, bank_data.y_train)
ada_untuned_learning_curve_plot = LearningCurvePlot(experiment=ada_untuned_learning_curve)

ada_tuned_learning_curve = LearningCurve(ada_tuned_clf, scoring='recall')
ada_tuned_learning_curve.run(bank_data.X_train, bank_data.y_train)
ada_tuned_learning_curve_plot = LearningCurvePlot(experiment=ada_tuned_learning_curve)

ada_tuned_cv = CrossValidation(ada_tuned_clf, scoring='recall')
ada_tuned_cv.run(bank_data.X_train, bank_data.y_train)
ada_tuned_cv_folds = CrossValidationFolds(experiment=ada_tuned_cv)

boost_iter = BoostingIterations(ada_tuned_clf, scoring='recall')
boost_iter.run(bank_data.X_train, bank_data.y_train)
boost_iter_plot = BoostingIterationsPlot(experiment=boost_iter)

ada_train_curv_fig, axs = plt.subplots(2, 2, figsize=(18, 14))
_, ax = ada_untuned_learning_curve_plot.plot(ax=axs[0][0], ymin=0.0, ymax=1.02)
_, ax = ada_tuned_learning_curve_plot.plot(ax=axs[0][1], ymin=0.0, ymax=1.02)
_, ax = ada_tuned_cv_folds.plot(title="Tuned model", ax=axs[1][0])
_, ax = boost_iter_plot.plot(ax=axs[1][1])

ada_train_curv_fig.savefig(
    os.path.join(out_dir, "BankData", "AdaBoost", "Learning Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

dt = DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=1500)
ada_clf = AdaBoostClassifier(base_estimator=dt, learning_rate=0.0001)

ada_val_curv_A = ValidationCurve(
    ada_clf,
    'n_estimators',
    np.arange(100, 600, 100),
    scoring='recall',
    n_splits=3
)
ada_val_curv_A.run(bank_data.X_train, bank_data.y_train)
ada_val_curv_A_plot = ValidationCurvePlot(experiment=ada_val_curv_A)

dt = DecisionTreeClassifier(class_weight='balanced', max_depth=1)
ada_clf = AdaBoostClassifier(base_estimator=dt, learning_rate=1.0)

ada_val_curv_B = ValidationCurve(
    ada_clf,
    'n_estimators',
    np.arange(100, 600, 100),
    scoring='recall',
    n_splits=3
)
ada_val_curv_B.run(bank_data.X_train, bank_data.y_train)
ada_val_curv_B_plot = ValidationCurvePlot(experiment=ada_val_curv_B)

ada_val_curvs_fig, axs = plt.subplots(1, 2, figsize=(10, 4))
_, ax = ada_val_curv_A_plot.plot(ax=axs[0], title="min_samples_leaf=1500, lr=0.0001")
_, ax = ada_val_curv_B_plot.plot(ax=axs[1], title="max_depth=1, lr=1.0")

ada_val_curvs_fig.savefig(
    os.path.join(out_dir, "BankData", "AdaBoost", "Validation Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

ada_tuned_model_eval = ModelEvaluation(ada_tuned_clf)
ada_tuned_model_eval.run(bank_data.X_train, bank_data.y_train, bank_data.X_test, bank_data.y_test)
print(ada_tuned_model_eval.model)
print(ClassificationReport(experiment=ada_tuned_model_eval).plot())

######################
### Neural Network ###
######################
nn_untuned_clf = NeuralNetworkClassifier(
    n_layers=2,
    in_features=bank_data.X_train.shape[1],
    out_features=26,
    epochs=10,
    activation_fn=nn.ReLU,
    learning_rate=0.1,
    verbose=True
)
nn_tuned_clf = NeuralNetworkClassifier(
    n_layers=2,
    in_features=bank_data.X_train.shape[1],
    out_features=26,
    epochs=20,
    activation_fn=nn.ReLU,
    learning_rate=0.4,
    verbose=True
)

nn_untuned_learning_curve = LearningCurve(nn_untuned_clf, scoring='recall', n_splits=3)
nn_untuned_learning_curve.run(bank_data.X_train, bank_data.y_train)
nn_untuned_learning_curve_plot = LearningCurvePlot(experiment=nn_untuned_learning_curve)

nn_tuned_learning_curve = LearningCurve(nn_tuned_clf, scoring='recall', n_splits=3)
nn_tuned_learning_curve.run(bank_data.X_train, bank_data.y_train)
nn_tuned_learning_curve_plot = LearningCurvePlot(experiment=nn_tuned_learning_curve)

nn_tuned_cv = CrossValidation(nn_tuned_clf, scoring='recall', n_splits=3)
nn_tuned_cv.run(bank_data.X_train, bank_data.y_train)
nn_tuned_cv_folds = CrossValidationFolds(experiment=nn_tuned_cv)

epochs_nn_clf = NeuralNetworkClassifier(
    n_layers=2,
    in_features=bank_data.X_train.shape[1],
    out_features=50,
    epochs=30,
    verbose=True,
    learning_rate=0.0001,
    activation_fn=nn.ELU
)
tuned_learning_curve_epochs = LearningCurveEpochs(
    epochs_nn_clf,
    scoring='cross-entropy',
    n_splits=3
)
tuned_learning_curve_epochs.run(bank_data.X_train, bank_data.y_train)
tuned_learning_curve_epochs_plot = LearningEpochsCurvePlot(experiment=tuned_learning_curve_epochs)

nn_train_curv_fig, axs = plt.subplots(1, 4, figsize=(24, 5))
_, ax = nn_untuned_learning_curve_plot.plot(ax=axs[0], title="epochs=10, layer_size=26, lr=0.1, ReLU", ymin=0.0, ymax=1.02)
_, ax = nn_tuned_learning_curve_plot.plot(ax=axs[1], title="epochs=20, layer_size=26, lr=0.4, ReLU", ymin=0.0, ymax=1.02)
_, ax = nn_tuned_cv_folds.plot(ax=axs[2], title="epochs=20, layer_size=26, lr=0.4, ReLU")
_, ax = tuned_learning_curve_epochs_plot.plot(ax=axs[3])

nn_train_curv_fig.savefig(
    os.path.join(out_dir, "BankData", "NeuralNetwork", "Learning Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

nn_clf = NeuralNetworkClassifier(
    n_layers=2,
    in_features=bank_data.X_train.shape[1],
    out_features=25,
    epochs=8,
    activation_fn=nn.ReLU,
    learning_rate=0.4,
    verbose=True
)
nn_val_curv_relu_lr = ValidationCurve(
    nn_clf,
    'learning_rate',
    [0.1, 0.2, 0.3, 0.4, 0.5],
    scoring='recall',
    n_splits=3,
)
nn_val_curv_relu_lr.run(bank_data.X_train, bank_data.y_train)
nn_val_curv_relu_lr_plot = ValidationCurvePlot(experiment=nn_val_curv_relu_lr)

nn_clf = NeuralNetworkClassifier(
    n_layers=2,
    in_features=bank_data.X_train.shape[1],
    out_features=20,
    epochs=8,
    activation_fn=nn.ReLU,
    learning_rate=0.4,
    verbose=True
)
nn_val_curv_relu = ValidationCurve(
    nn_clf,
    'out_features',
    [25, 50, 100],
    scoring='recall',
    n_splits=3,
)
nn_val_curv_relu.run(bank_data.X_train, bank_data.y_train)
nn_val_curv_relu_plot = ValidationCurvePlot(experiment=nn_val_curv_relu)

nn_clf = NeuralNetworkClassifier(
    n_layers=2,
    in_features=bank_data.X_train.shape[1],
    out_features=20,
    epochs=20,
    activation_fn=nn.ELU,
    learning_rate=0.2,
    verbose=True
)
nn_val_curv_elu = ValidationCurve(
    nn_clf,
    'out_features',
    [25, 75, 125],
    scoring='recall',
    n_splits=3
)
nn_val_curv_elu.run(bank_data.X_train, bank_data.y_train)
nn_val_curv_elu_plot = ValidationCurvePlot(experiment=nn_val_curv_elu)

nn_val_curvs_fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
_, ax = nn_val_curv_relu_lr_plot.plot(ax=axs[0], title="epochs=8, layer_size=25")
_, ax = nn_val_curv_relu_plot.plot(ax=axs[1], title="epochs=8, lr=0.4")
_, ax = nn_val_curv_elu_plot.plot(ax=axs[2], title="activation_fn=ELU, lr=0.2")

nn_val_curvs_fig.savefig(
    os.path.join(out_dir, "BankData", "NeuralNetwork", "Validation Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

nn_tuned_model_eval = ModelEvaluation(nn_tuned_clf)
nn_tuned_model_eval.run(bank_data.X_train, bank_data.y_train, bank_data.X_test, bank_data.y_test)
print(nn_tuned_model_eval.model)
print(ClassificationReport(experiment=nn_tuned_model_eval).plot())

########################
### Wall Clock Times ###
########################

compare_fit_scores_times_plot = CompareFitScoreTimes(experiments=[
    dt_tuned_learning_curve,
    svm_tuned_learning_curve,
    knn_tuned_learning_curve,
    ada_tuned_learning_curve,
    nn_tuned_learning_curve
])

compare_fit_scores_times_fig, ax = compare_fit_scores_times_plot.plot(figsize=(16, 6), ymin=0.0, ymax=8.0)

compare_fit_scores_times_fig.savefig(
    os.join(out_dir, "BankData", "Compare Fit and Times - Report.png"),
    bbox_inches='tight',
    dpi=800
)