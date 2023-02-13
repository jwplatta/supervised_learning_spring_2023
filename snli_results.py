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
data_files = [
  os.path.join(working_dir, "data", "snli_word2vec.model"),
  os.path.join(working_dir, "data", "snli_1.0_train.jsonl"),
  os.path.join(working_dir, "data", "snli_1.0_test.jsonl")
]


if not os.path.exists(out_dir):
    os.mkdir(out_dir)


for df in data_files:
  if not os.path.exists(df):
      raise Exception("Please get the data from the dropbox link in the readme.txt: {0}".format(df))

plt.style.use('seaborn')

random_state = 279
snli_data = snli_data = SNLIData(random_state=random_state)

X_train, X_val, y_train, y_val = train_test_split(
    snli_data.X_train,
    snli_data.y_train,
    train_size=100000,
    test_size=100000,
    random_state=random_state
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_val = X_val.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

def recall_entailment_contradiction(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[0, 1], average='macro')

recall_scorer = make_scorer(recall_entailment_contradiction)

X_train_samp, _, y_train_samp, _ = train_test_split(X_train, y_train, train_size=10000, random_state=random_state)
X_train_samp = X_train_samp.reset_index(drop=True)
y_train_samp = y_train_samp.reset_index(drop=True)


########################
### Data Exploration ###
########################

class_balance_plot = Histogram(
    y_train,
    labels=y_train,
    xticks=snli_data.classes,
    title="Target Class Balance"
)

ft81_hist_plot = Histogram(
    X_train['feature_81'],
    feature_name='feature_81',
    labels=y_train,
    bins=100,
    label_names=snli_data.classes,
    legend=True
)

ft9_hist_plot = Histogram(
    X_train['feature_9'],
    feature_name='feature_9',
    labels=y_train,
    bins=100,
    label_names=snli_data.classes,
    legend=True
)

contr_entail_y_train_idx = y_train[(y_train == 0) | (y_train == 1)].index
contra_entail_instances = X_train.iloc[contr_entail_y_train_idx].copy()
contra_entail_instances['label'] = y_train[contr_entail_y_train_idx].copy()

compare_ft81_to_ft9_scatter = ScatterPlot(
    contra_entail_instances,
    'feature_9', 'feature_81', 'label', label_names=['contradiction', 'entailment'])

data_exp_fig, axs = plt.subplots(1, 4, figsize=(19, 4))
_, ax = class_balance_plot.plot(ax=axs[0])
_, ax = ft81_hist_plot.plot(ax=axs[1])
_, ax = ft9_hist_plot.plot(ax=axs[2])
_, ax = compare_ft81_to_ft9_scatter.plot(ax=axs[3])

data_exp_fig.savefig(
    os.path.join(out_dir, "SNLI", "Data Exploration - Report.png"),
    bbox_inches='tight',
    dpi=800
)


#####################
### Decision Tree ###
#####################
untuned_dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2
)

tuned_dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=7,
    min_samples_leaf=1,
    min_samples_split=1925
)

dt_untuned_learning_curve = LearningCurve(untuned_dt_clf, scoring=recall_scorer)
dt_untuned_learning_curve.run(X_train, y_train)
dt_untuned_learning_curve_plot = LearningCurvePlot(experiment=dt_untuned_learning_curve)

dt_tuned_learning_curve = LearningCurve(tuned_dt_clf, scoring=recall_scorer)
dt_tuned_learning_curve.run(X_train, y_train)
dt_tuned_learning_curve_plot = LearningCurvePlot(experiment=dt_tuned_learning_curve)

dt_tuned_cv = CrossValidation(tuned_dt_clf, scoring=recall_scorer)
dt_tuned_cv.run(X_train, y_train)
dt_tuned_cv_folds = CrossValidationFolds(experiment=dt_tuned_cv)

dt_train_curv_fig, axs = plt.subplots(1, 3, figsize=(20, 5))
_, ax = dt_untuned_learning_curve_plot.plot(ax=axs[0], ymin=0.0, ymax=1.01)
_, ax = dt_tuned_learning_curve_plot.plot(ax=axs[1], ymin=0.0, ymax=1.01, title="max_depth=7, min_samples_split=1925")
_, ax = dt_tuned_cv_folds.plot(ax=axs[2])

dt_train_curv_fig.savefig(
    os.path.join(out_dir, "SNLI", "DecisionTree", "LearningCurves - Report.png")
    bbox_inches='tight',
    dpi=800
)

val_curv = ValidationCurve(
    DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'),
    'min_samples_leaf',
   np.arange(1, 3500, 500),
    scoring=recall_scorer
)
val_curv.run(X_train, y_train)
min_samples_leaf_plot = ValidationCurvePlot(experiment=val_curv)

val_curv = ValidationCurve(
    DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'),
    'max_depth',
    np.arange(2, 15, 2),
    scoring=recall_scorer
)
val_curv.run(X_train, y_train)
max_depth_plot = ValidationCurvePlot(experiment=val_curv)

val_curv = ValidationCurve(
    DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'),
    'min_samples_split',
    np.arange(2, 3500, 500),
    scoring=recall_scorer
)
val_curv.run(X_train, y_train)
min_samples_split_plot = ValidationCurvePlot(experiment=val_curv)

validation_fig, axs = plt.subplots(1, 3, figsize=(16, 4))
_, ax = max_depth_plot.plot(ax=axs[0])
_, ax = min_samples_split_plot.plot(ax=axs[1])
_, ax = min_samples_leaf_plot.plot(ax=axs[2])

validation_fig.savefig(
    os.path.join(out_dir, "SNLI", "DecisionTree", "ValidationCurves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

tuned_model_eval = ModelEvaluation(tuned_dt_clf)
tuned_model_eval.run(X_train, y_train, snli_data.X_test, snli_data.y_test)
print(tuned_model_eval.model)
print(ClassificationReport(experiment=tuned_model_eval).plot())
##############################
### Support Vector Machine ###
##############################

untuned_svm_clf = LinearSVC(
    C=1.0, max_iter=1000, penalty='l2', dual=False, verbose=0
)

# SVC(C=0.15, gamma=0.01, kernel='rbf')
tuned_svm_clf = SVC(C=0.05, gamma=0.01, kernel='rbf')

svm_untuned_learning_curve = LearningCurve(
    untuned_svm_clf,
    scoring=recall_scorer,
    n_splits=3
)
svm_untuned_learning_curve.run(X_train, y_train)

svm_tuned_learning_curve = LearningCurve(
    tuned_svm_clf,
    scoring=recall_scorer,
    n_splits=3
)
svm_tuned_learning_curve.run(X_train_sub, y_train_sub)

svm_tuned_cv = CrossValidation(tuned_svm_clf, scoring=recall_scorer)
svm_tuned_cv.run(X_train_sub, y_train_sub)
svm_tuned_cv_folds = CrossValidationFolds(experiment=svm_tuned_cv)

svm_untuned_learning_curve_plot = LearningCurvePlot(experiment=svm_untuned_learning_curve)
svm_tuned_learning_curve_plot = LearningCurvePlot(experiment=svm_tuned_learning_curve)

svm_learning_curves_fig, axs = plt.subplots(1, 3, figsize=(18, 5))
_, ax = svm_untuned_learning_curve_plot.plot(ax=axs[0], ymin=0.0, ymax=1.01)
_, ax = svm_tuned_learning_curve_plot.plot(ax=axs[1], ymin=0.0, ymax=1.01)
_, ax = svm_tuned_cv_folds.plot(ax=axs[2])

svm_learning_curves_fig.savefig(
    os.path.join(out_dir, "SNLI", "SVM", "LearningCurves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

linear_svc_val_curv = ValidationCurve(
    LinearSVC(C=1.0, max_iter=1000, penalty='l2', dual=False, verbose=0),
    'C',
    [0.001, 0.01, 0.1, 1.0, 10],
    scoring=recall_scorer,
    n_splits=3
)
linear_svc_val_curv.run(X_train, y_train)

svc_high_gamma_val_curv = ValidationCurve(
    SVC(gamma=0.01, kernel='rbf', verbose=0),
    'C',
    np.arange(0.01, 4, 1.0),
    scoring=recall_scorer,
    n_splits=3
)
svc_high_gamma_val_curv.run(X_train_sub, y_train_sub)

svc_low_gamma_val_curv = ValidationCurve(
    SVC(gamma=0.0001, kernel='rbf', verbose=0),
    'C',
    np.arange(0.01, 4, 1.0),
    scoring=recall_scorer,
    n_splits=3
)
svc_low_gamma_val_curv.run(X_train_samp, y_train_sub)

svc_gamma_val_curv = ValidationCurve(
    SVC(C=0.1, kernel='rbf', verbose=0),
    'gamma',
    [0.001, 0.005, 0.01, 0.05, 0.1],
    scoring=recall_scorer,
    n_splits=3
)
svc_gamma_val_curv.run(X_train_sub, y_train_sub)

svc_gamma_val_curv_plot = ValidationCurvePlot(experiment=svc_gamma_val_curv)
linear_svc_val_curv_plot = ValidationCurvePlot(experiment=linear_svc_val_curv)
svc_high_gamma_val_curv_plot = ValidationCurvePlot(experiment=svc_high_gamma_val_curv)
svc_low_gamma_val_curv_plot = ValidationCurvePlot(experiment=svc_low_gamma_val_curv)

svc_val_curvs_fig, axs = plt.subplots(2, 2, figsize=(14, 11))
_, ax = linear_svc_val_curv_plot.plot(ax=axs[0][0])
_, ax = svc_high_gamma_val_curv_plot.plot(ax=axs[0][1])
_, ax = svc_gamma_val_curv_plot.plot(ax=axs[1][0])
_, ax = svc_low_gamma_val_curv_plot.plot(ax=axs[1][1])

svc_val_curvs_fig.savefig(
    os.path.join(out_dir, "SNLI", "SVM", "Validation Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

tuned_model_eval = ModelEvaluation(tuned_svm_clf)
tuned_model_eval.run(X_train, y_train, snli_data.X_test, snli_data.y_test)
print(tuned_model_eval.model)
print(ClassificationReport(experiment=tuned_model_eval).plot())

###########################
### k-Nearest Neighbors ###
###########################
knn_untuned_clf = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform')
knn_tuned_clf = KNeighborsClassifier(n_neighbors=9, weights='uniform', p=2)

knn_tuned_cv = CrossValidation(knn_tuned_clf, scoring=recall_scorer, n_splits=3)
knn_tuned_cv.run(X_train, y_train)
knn_tuned_cv_folds = CrossValidationFolds(experiment=knn_tuned_cv)

knn_untuned_learning_curve = LearningCurve(knn_untuned_clf, scoring=recall_scorer, n_splits=3)
knn_untuned_learning_curve.run(X_train, y_train)
knn_untuned_learning_curve_plot = LearningCurvePlot(experiment=knn_untuned_learning_curve)

knn_tuned_learning_curve = LearningCurve(knn_tuned_clf, scoring=recall_scorer, n_splits=3)
knn_tuned_learning_curve.run(X_train, y_train)
knn_tuned_learning_curve_plot = LearningCurvePlot(experiment=knn_tuned_learning_curve)

knn_train_curv_fig, axs = plt.subplots(1, 3, figsize=(16, 4))
_, ax = knn_untuned_learning_curve_plot.plot(ax=axs[0], ymin=0.0, ymax=1.02)
_, ax = knn_tuned_learning_curve_plot.plot(ax=axs[1], ymin=0.0, ymax=1.02)
_, ax = knn_tuned_cv_folds.plot(ax=axs[2])

knn_train_curv_fig.savefig(
    os.path.join(out_dir, "SNLI", "KNN", "Training Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

distance_val_curv = ValidationCurve(
    KNeighborsClassifier(p=2, weights='distance', leaf_size=5000, algorithm='ball_tree'),
    'n_neighbors',
    [100, 5000, 10000],
    scoring=recall_scorer,
    n_splits=3
)
distance_val_curv.run(X_train_sub, y_train_sub)

uniform_val_curv = ValidationCurve(
    KNeighborsClassifier(p=1, weights='uniform'),
    'n_neighbors',
    np.arange(2, 120, 20),
    scoring=recall_scorer,
    n_splits=3
)
uniform_val_curv.run(X_train_sub, y_train_sub)

distance_val_curv_plot =  ValidationCurvePlot(experiment=distance_val_curv)
uniform_val_curv_plot =  ValidationCurvePlot(experiment=uniform_val_curv)

knn_val_curvs_fig, axs = plt.subplots(1, 2, figsize=(12, 5))
_, ax = distance_val_curv_plot.plot(ax=axs[0])
_, ax = uniform_val_curv_plot.plot(ax=axs[1])

knn_val_curvs_fig.savefig(
    os.path.join(out_dir, "SNLI", "KNN", "Validation Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

knn_tuned_model_eval = ModelEvaluation(knn_tuned_clf)
knn_tuned_model_eval.run(X_train, y_train, snli_data.X_test, snli_data.y_test)
print(knn_tuned_model_eval.model)
print(ClassificationReport(experiment=knn_tuned_model_eval).plot())

################
### AdaBoost ###
################

dstump = DecisionTreeClassifier(max_depth=1)
ada_untuned_clf = AdaBoostClassifier(base_estimator=dstump, n_estimators=200, learning_rate=0.1)
weak_learner = DecisionTreeClassifier(max_depth=2)
ada_tuned_clf = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=250, learning_rate=0.05)

ada_untuned_learning_curve = LearningCurve(ada_untuned_clf, scoring=recall_scorer, n_splits=3)
ada_untuned_learning_curve.run(X_train, y_train)
ada_untuned_learning_curve_plot = LearningCurvePlot(experiment=ada_untuned_learning_curve)

ada_tuned_learning_curve = LearningCurve(ada_tuned_clf, scoring=recall_scorer, n_splits=3)
ada_tuned_learning_curve.run(X_train, y_train)
ada_tuned_learning_curve_plot = LearningCurvePlot(experiment=ada_tuned_learning_curve)

ada_tuned_cv = CrossValidation(ada_tuned_clf, scoring=recall_scorer, n_splits=3)
ada_tuned_cv.run(X_train_samp, y_train_samp)
ada_tuned_cv_folds = CrossValidationFolds(experiment=ada_tuned_cv)

boost_iter = BoostingIterations(ada_tuned_clf, n_splits=3)
boost_iter.run(X_train_samp, y_train_samp)
boost_iter_plot = BoostingIterationsPlot(experiment=boost_iter)

ada_train_curv_fig, axs = plt.subplots(1, 3, figsize=(15, 5))
_, ax = ada_untuned_learning_curve_plot.plot(ax=axs[0], ymin=0.0, ymax=1.02, title="lr=0.1, n_estimators=200, max_depth=1")
_, ax = ada_tuned_learning_curve_plot.plot(ax=axs[1], ymin=0.0, ymax=1.02, title="lr=0.05, n_estimators=250, max_depth=2")
# _, ax = ada_tuned_cv_folds.plot(title="Tuned model", ax=axs[1][0])
_, ax = boost_iter_plot.plot(ax=axs[2])

ada_train_curv_fig.savefig(
    os.path.join(out_dir, "SNLI", "AdaBoost", "Learning Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

dt = DecisionTreeClassifier(max_depth=3)
ada_clf = AdaBoostClassifier(base_estimator=dt, learning_rate=0.1)

high_complexity_high_lr_curv = ValidationCurve(
    ada_clf,
    'n_estimators',
    np.arange(100, 500, 100),
    scoring=recall_scorer,
    n_splits=3
)
high_complexity_high_lr_curv.run(X_train_samp, y_train_samp)
high_complexity_high_lr_curv_plot = ValidationCurvePlot(experiment=high_complexity_high_lr_curv)

dt = DecisionTreeClassifier(max_depth=3)
ada_clf = AdaBoostClassifier(base_estimator=dt, learning_rate=0.01)

high_complexity_low_lr_curv = ValidationCurve(
    ada_clf,
    'n_estimators',
    np.arange(100, 500, 100),
    scoring=recall_scorer,
    n_splits=3
)
high_complexity_low_lr_curv.run(X_train_samp, y_train_samp)
high_complexity_low_lr_curv_plot = ValidationCurvePlot(experiment=high_complexity_low_lr_curv)

ada_val_curvs_fig, axs = plt.subplots(1, 2, figsize=(10, 4))
_, ax = high_complexity_high_lr_curv_plot.plot(ax=axs[0], title="max_depth=3, lr=0.1")
_, ax = high_complexity_low_lr_curv_plot.plot(ax=axs[1],  title="max_depth=3, lr=0.01")

ada_val_curvs_fig.savefig(
    os.path.join(out_dir, "SNLI", "AdaBoost", "Validation Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

ada_tuned_model_eval = ModelEvaluation(ada_tuned_clf)
ada_tuned_model_eval.run(X_train, y_train, snli_data.X_test, snli_data.y_test)
print(ada_tuned_model_eval.model)
print(ClassificationReport(experiment=ada_tuned_model_eval).plot())


#####################
### NeuralNetwork ###
#####################
nn_untuned_clf = NeuralNetworkClassifier(
    n_layers=2,
    in_features=snli_data.X_train.shape[1],
    out_features=150,
    epochs=20,
    learning_rate=0.1,
    activation_fn=nn.ReLU,
    verbose=True
)
nn_tuned_clf = NeuralNetworkClassifier(
    n_layers=2,
    in_features=snli_data.X_train.shape[1],
    out_features=100,
    epochs=60,
    activation_fn=nn.ReLU,
    learning_rate=0.005,
    verbose=True
)

nn_untuned_learning_curve = LearningCurve(nn_untuned_clf, scoring=recall_scorer, n_splits=2)
nn_untuned_learning_curve.run(X_train_samp, y_train_samp)
nn_untuned_learning_curve_plot = LearningCurvePlot(experiment=nn_untuned_learning_curve)

nn_tuned_learning_curve = LearningCurve(nn_tuned_clf, scoring=recall_scorer, n_splits=2)
nn_tuned_learning_curve.run(X_train_samp, y_train_samp)
nn_tuned_learning_curve_plot = LearningCurvePlot(experiment=nn_tuned_learning_curve)

nn_tuned_cv = CrossValidation(nn_tuned_clf, scoring=recall_scorer, n_splits=3)
nn_tuned_cv.run(X_train_samp, y_train_samp)
nn_tuned_cv_folds = CrossValidationFolds(experiment=nn_tuned_cv)

nn_clf = NeuralNetworkClassifier(
    epochs=100,
    n_layers=2,
    out_features=600,
    learning_rate=0.00009,
    in_features=X_train.shape[1],
    activation_fn=nn.ReLU,
    verbose=True
)
tuned_learning_curve_epochs = LearningCurveEpochs(
    nn_clf,
    scoring='cross-entropy',
    n_splits=3
)
tuned_learning_curve_epochs.run(X_train_samp, y_train_samp)
tuned_learning_curve_epochs_plot = LearningEpochsCurvePlot(experiment=tuned_learning_curve_epochs)

nn_train_curv_fig, axs = plt.subplots(1, 3, figsize=(16, 4))
_, ax = nn_untuned_learning_curve_plot.plot(ax=axs[0], ymin=0.0, ymax=1.02)
_, ax = nn_tuned_learning_curve_plot.plot(ax=axs[1], ymin=0.0, ymax=1.02)
# _, ax = nn_tuned_cv_folds.plot(ax=axs[2])
_, ax = tuned_learning_curve_epochs_plot.plot(ax=axs[2])

nn_train_curv_fig.savefig(
    os.path.join(out_dir, "SNLI", "NeuralNetwork", "Learning Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

nn_nodes_val_curv = ValidationCurve(
    NeuralNetworkClassifier(
        epochs=60,
        n_layers=2,
        learning_rate=0.001,
        in_features=X_train.shape[1],
        activation_fn=nn.ReLU,
        verbose=True
    ),
    'out_features',
    [25, 50, 100, 200],
    scoring=recall_scorer,
    n_splits=2
)
nn_nodes_val_curv.run(X_train_samp, y_train_samp)
nn_nodes_val_curv_plot = ValidationCurvePlot(experiment=nn_nodes_val_curv)

nn_lr_val_curv = ValidationCurve(
    NeuralNetworkClassifier(
        epochs=40,
        n_layers=2,
        in_features=X_train.shape[1],
        activation_fn=nn.ReLU,
        out_features=100,
        verbose=True
    ),
    'learning_rate',
    [0.0001, 0.001, 0.01, 0.1],
    scoring=recall_scorer,
    n_splits=2
)
nn_lr_val_curv.run(X_train_samp, y_train_samp)
nn_lr_val_curv_plot = ValidationCurvePlot(experiment=nn_lr_val_curv)

nn_val_curvs_fig, axs = plt.subplots(1, 2, figsize=(10, 4))
_, ax = nn_nodes_val_curv_plot.plot(ax=axs[0])
_, ax = nn_lr_val_curv_plot.plot(ax=axs[1])

nn_val_curvs_fig.savefig(
    os.path.join(out_dir, "SNLI", "NeuralNetwork", "Validation Curves - Report.png"),
    bbox_inches='tight',
    dpi=800
)

nn_tuned_model_eval = ModelEvaluation(nn_tuned_clf)
nn_tuned_model_eval.run(X_train[:10000], y_train[:10000], snli_data.X_test, snli_data.y_test)
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

compare_fit_scores_times_fig, ax = compare_fit_scores_times_plot.plot(figsize=(16, 6))

compare_fit_scores_times_fig.savefig(
    os.path.join(out_dir, "SNLI", "Compare Fit and Times - Report.png"),
    bbox_inches='tight',
    dpi=800
)