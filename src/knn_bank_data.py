import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from datasets.BankData import BankData

from experiments.ValidationCurve import ValidationCurve
from plots.ValidationCurve import ValidationCurve as ValidationCurvePlot

untuned_clf = knn_clf = KNeighborsClassifier(n_neighbors=8, weights='uniform', p=2)
intermediate_clf = None
tuned_clf = None


def learning_curve_plots():
    pass


def model_evaluation_plots():
    pass


def validation_curve_plots(bank_data):
    uniform_val_curv = ValidationCurve(
        KNeighborsClassifier(n_neighbors=8, weights='uniform', p=2),
        'n_neighbors',
        [2, 4, 6, 8, 10, 12],
        scoring='recall'
    )
    uniform_val_curv.run(bank_data.X_train, bank_data.y_train)
    uniform_val_curv_plot = ValidationCurvePlot(experiment=uniform_val_curv)

    distance_val_curv = ValidationCurve(
        KNeighborsClassifier(n_neighbors=8, weights='distance', p=2),
        'n_neighbors',
        [2, 4, 6, 8, 10, 12],
        scoring='recall'
    )
    distance_val_curv.run(bank_data.X_train, bank_data.y_train)
    distance_val_curv_plot = ValidationCurvePlot(experiment=distance_val_curv)

    val_curvs_fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    _, ax = uniform_val_curv_plot.plot(ax=axs[0], title="Validation Curve for n Neighbors (Uniform Weights)")
    _, ax = distance_val_curv_plot.plot(ax=axs[1], title="Validation Curve for n Neighbors (Distance Weights)")

    val_curvs_fig.savefig(
        "../out/BankData/KNN/ValidationCurvesComparison.png",
        bbox_inches='tight',
        dpi=800
    )

    return True


def test_data_results():
    pass


def generate_results():
    print('generating results')
    random_state = 42
    bank_data = BankData()
    validation_curve_plots(bank_data)

    # STEP: Learning Curves

    # STEP : Confusion Matrices

    # STEP: Save Classification Reports

    # STEP: Validation Curves
    validation_curve_plots(bank_data)

    # STEP: Final Results



if __name__ == '__main__':
    generate_results()

