
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import train_test_split

from datasets.SNLIData import SNLIData

from experiments.LearningCurve import LearningCurve
from experiments.ValidationCurve import ValidationCurve

from plots.LearningCurve import LearningCurve as LearningCurvePlot
from plots.ValidationCurve import ValidationCurve as ValidationCurvePlot


import matplotlib.pyplot as plt
import numpy as np


untuned_dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1
)


intermediate_dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10,
    min_samples_leaf=625,
    min_samples_split=2
) # NOTE: best so far at 40%


tuned_dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=7,
    min_samples_leaf=1,
    min_samples_split=1925,
    class_weight={0: 1.05, 1: 1.0, 2: 0.93}
)
# DecisionTreeClassifier(max_depth=7, min_samples_split=1925, class_weight={0: 1.05, 1:1.0, 2: 0.98})
# DecisionTreeClassifier(class_weight={0: 1.0, 1: 1.0, 2: 0.9}, max_depth=7, min_samples_split=1925)


def recall_scorer():
    def recall_classes_0_and_1(y_true, y_pred):
        return recall_score(y_true, y_pred, labels=[0, 1], average='micro')

    return make_scorer(recall_classes_0_and_1)


def learning_curve_plots(X_train, y_train, random_state):
    untuned_learning_curve = LearningCurve(
        untuned_dt_clf,
        random_state=random_state,
        scoring='accuracy',
    )
    untuned_learning_curve.run(X_train, y_train)

    intermediate_learning_curve = LearningCurve(
        intermediate_dt_clf,
        random_state=random_state,
        scoring='accuracy'
    )
    intermediate_learning_curve.run(X_train, y_train)

    tuned_learning_curve = LearningCurve(
        tuned_dt_clf,
        random_state=random_state,
        scoring='accuracy'
    )
    tuned_learning_curve.run(X_train, y_train)

    train_curvs_fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    _, ax = LearningCurvePlot(experiment=untuned_learning_curve).plot(ax=axs[0], title="Learning Curve (Untuned Model)")
    _, ax = LearningCurvePlot(experiment=intermediate_learning_curve).plot(ax=axs[1], title="Learning Curve (Intermediate Model)")
    _, ax = LearningCurvePlot(experiment=tuned_learning_curve).plot(ax=axs[2], title="Learning Curve (Tuned Model)")

    train_curvs_fig.savefig(
        "../out/SNLI/DecisionTree/LearningCurves.png",
        bbox_inches='tight',
        dpi=800
    )

    return True


def validation_curve_plots(X_train, y_train, random_state):
    val_curv_max_depth_accuracy = ValidationCurve(
        DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=2),
        'max_depth',
        np.arange(2, 14, 2),
        scoring='accuracy',
        random_state=random_state
    )
    val_curv_max_depth_accuracy.run(X_train, y_train)
    val_curv_max_depth_accuracy_plot = ValidationCurvePlot(experiment=val_curv_max_depth_accuracy)

    val_curv_leaf_recall = ValidationCurve(
        DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=2),
        'min_samples_leaf',
        np.arange(1, 4502, 500),
        scoring=recall_scorer,
        random_state=random_state
    )
    val_curv_leaf_recall.run(X_train, y_train)
    val_curv_leaf_recall_plot = ValidationCurvePlot(experiment=val_curv_leaf_recall)

    val_curv_split_recall = ValidationCurve(
    DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=1),
        'min_samples_split',
        np.arange(2, 7003, 1000),
        n_splits=3,
        scoring=recall_scorer,
        random_state=random_state
    )
    val_curv_split_recall.run(X_train, y_train)
    val_curv_split_recall_plot = ValidationCurvePlot(experiment=val_curv_split_recall)

    val_curv_max_depth_recall = ValidationCurve(
        DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1000, min_samples_split=1500),
        'max_depth',
        np.arange(2, 14, 2),
        scoring=recall_scorer
    )
    val_curv_max_depth_recall.run(X_train, y_train)
    val_curv_max_depth_recall_plot = ValidationCurvePlot(experiment=val_curv_max_depth_recall)

    val_curvs_fig, axs = plt.subplots(2, 2, figsize=(10, 4))
    _, ax = val_curv_max_depth_accuracy_plot.plot(ax=axs[0][0], title="Validation Curve - Accuracy (Max Depth)")
    _, ax = val_curv_leaf_recall_plot.plot(ax=axs[0][1], title="Validation Curve - Recall (Leaf Size)")
    _, ax = val_curv_split_recall_plot.plot(ax=axs[1][0], title="Validation Curve - Recall (Split Size)")
    _, ax = val_curv_max_depth_recall_plot.plot(ax=axs[1][1], title="Validation Curve - Recall (Max Depth)")

    val_curvs_fig.savefig(
        "../out/SNLI/DecisionTree/ValidationCurves.png",
        bbox_inches='tight',
        dpi=800
    )

    return True


def model_evaluation_plots():
    pass


def test_data_results():
    pass


def generate_results():
    print('Generating results for Decision Tree SNLI')
    random_state = 279

    snli_data = SNLIData()

    X_train, X_val, y_train, y_val = train_test_split(
        snli_data.X_train,
        snli_data.y_train,
        train_size=60000,
        test_size=60000,
        random_state=random_state
    )

    # STEP: Learning Curves
    learning_curve_plots(X_train, y_train, random_state)

    # STEP : Confusion Matrices

    # STEP: Save Classification Reports

    # STEP: Validation Curves
    # validation_curve_plots(X_train, y_train, random_state)

    # STEP: Final Results



if __name__ == '__main__':
    generate_results()

