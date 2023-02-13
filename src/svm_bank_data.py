import os

import matplotlib.pyplot as plt

from datasets.BankData import BankData

from sklearn.svm import SVC, LinearSVC

from experiments.LearningCurve import LearningCurve
from experiments.ValidationCurve import ValidationCurve

from plots.LearningCurve import LearningCurve as LearningCurvePlot
from plots.ValidationCurve import ValidationCurve as ValidationCurvePlot


# OR SVC(C=0.15, class_weight='balanced', gamma=0.15, verbose=0)
# OR SVC(C=0.2, class_weight='balanced', gamma=0.4, verbose=0) OR SVC(C=0.2, class_weight='balanced', gamma=0.6, verbose=0)
untuned_model = LinearSVC(C=1.0, max_iter=500, penalty='l2', dual=False, verbose=0, class_weight='balanced')
intermediate_model = SVC(C=0.17, class_weight='balanced', gamma=0.44, verbose=0)
tuned_model = SVC(C=0.13, class_weight='balanced', gamma=0.13, verbose=0) # NOTE: false positive rate is 19%
# NOTE: high false positive rate: SVC(C=0.15, class_weight='balanced', gamma=0.16, verbose=0) false positive rate is 18%
# However, they both round to 19% as nearest integer.

def learning_curve_plots(bank_data):
    try:
        untuned_learning_curve = LearningCurve(
            LinearSVC(C=1.0, max_iter=500, penalty='l2', dual=False, verbose=0, class_weight='balanced'),
            scoring='recall',
            n_splits=3
        )
        untuned_learning_curve.run(bank_data.X_train, bank_data.y_train)

        intermediate_learning_curve = LearningCurve(
            SVC(C=0.17, class_weight='balanced', gamma=0.44, verbose=0),
            scoring='recall',
            n_splits=3
        )
        intermediate_learning_curve.run(bank_data.X_train, bank_data.y_train)

        tuned_learning_curve = LearningCurve(
            SVC(C=0.13, class_weight='balanced', gamma=0.13, verbose=0),
            scoring='recall',
            n_splits=3
        )
        tuned_learning_curve.run(bank_data.X_train, bank_data.y_train)

        untuned_learn_curv_plot = LearningCurvePlot(experiment=untuned_learning_curve)
        intermediate_learn_curv_plot = LearningCurvePlot(experiment=intermediate_learning_curve)
        tuned_learn_curv_plot = LearningCurvePlot(experiment=tuned_learning_curve)

        learning_curves_fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        _, ax = untuned_learn_curv_plot.plot(ax=axs[0], title="Learning Curve (Untuned Model)")
        _, ax = intermediate_learn_curv_plot.plot(ax=axs[1], title="Learning Curve (Intermediate Model)")
        _, ax = tuned_learn_curv_plot.plot(ax=axs[2], title="Learning Curve (Final Model)")

        return True
    except Exception as e:
        print(e)
        return False


def model_evaluation_plots(bank_data):
    pass


def validation_curve_plots(bank_data):
  try:
      # NOTE: dataset too large for validation curves
      X_train, X_test, y_train, y_test = train_test_split(
          bank_data.X_train, bank_data.y_train, train_size=10000, random_state=42
      )

      # NOTE: Tune C for LinearSVC model
      lsvc_val_curv = ValidationCurve(
          LinearSVC(C=1.0, max_iter=500, penalty='l2', dual=False, verbose=0, class_weight='balanced'), # class_weight={0:0.1, 1:1.0}),
          'C',
          [0.00001, 0.0001, 0.001, 0.01, 0.1],
          scoring='recall'
      )
      lsvc_val_curv.run(X_train, y_train)
      lsvc_val_curv_plot = ValidationCurvePlot(experiment=lsvc_val_curv)

      # NOTE: Tune C for RBF kernel and reasonable gamma (0.1)
      svc_C_val_curv = ValidationCurve(
          SVC(C=0.1, gamma=0.1, kernel='rbf', class_weight='balanced', verbose=0),
          'C',
          [0.001, 0.01, 0.1, 1.0, 10],
          scoring='recall'
      )
      svc_C_val_curv.run(X_train, y_train)
      svc_C_val_curv_plot = ValidationCurvePlot(experiment=svc_C_val_curv)

      # NOTE: Tune gamma for RBF kernel and reasonable C (0.1)
      svc_gamma_val_curv = ValidationCurve(
          SVC(C=0.1, gamma='scale', kernel='rbf', class_weight='balanced', verbose=0),
          'gamma',
          [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0],
          scoring='recall'
      )
      svc_gamma_val_curv.run(X_train, y_train)
      svc_gamma_val_curv_plot = ValidationCurvePlot(experiment=svc_gamma_val_curv)

      val_curvs_fig, axs = plt.subplots(1, 3, figsize=(18, 5))
      _, ax = lsvc_val_curv_plot.plot(ax=axs[0], title="Validation Curve for LinearSVC C")
      _, ax = svc_C_val_curv_plot.plot(ax=axs[1], title="Validation Curve for RBF C")
      _, ax = svc_gamma_val_curv_plot.plot(ax=axs[2], title="Validation Curve for RBF gamma")

      val_curvs_fig.savefig(
          "../out/BankData/SVM/ValidationCurvesComparison.png",
          bbox_inches='tight',
          dpi=800
      )
      return True
  except Exception as e:
      print(e)
      return False


def test_data_results():
    pass


def generate_results():
    print('Generating results for SVM Bank Data')

    random_state = 42

    bank_data = BankData()
    # STEP: Learning Curves
    if not(learning_curve_plots(bank_data)):
        return

    # STEP : Confusion Matrices

    # STEP: Save Classification Reports

    # STEP: Validation Curves
    if not(validation_curve_plots(bank_data)):
        return

    # STEP: Final Results



if __name__ == '__main__':
    generate_results()

