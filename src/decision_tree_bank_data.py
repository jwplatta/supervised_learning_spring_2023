from datasets.BankData import BankData
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt



untuned_dt_clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced'
)

intermediate_dt_clf = DecisionTreeClassifier(
    class_weight='balanced',
    criterion='entropy',
    max_depth=8,
    min_samples_leaf=35,
    min_samples_split=2,
)

tuned_dt_clf = DecisionTreeClassifier(
    class_weight='balanced',
    criterion='entropy',
    max_depth=8,
    min_samples_leaf=50,
    min_samples_split=900
)
# DecisionTreeClassifier(class_weight='balanced', max_depth=9, min_samples_split=1100)


def learning_curve_plots():
    pass


def model_evaluation_plots():
    pass


def validation_curve_plots():
    pass


def test_data_results():
    pass


def generate_results():
    print('Generating results for Decision Tree Bank Data')
    random_state = 42
    # STEP: Learning Curves

    # STEP : Confusion Matrices

    # STEP: Save Classification Reports

    # STEP: Validation Curves

    # STEP: Final Results



if __name__ == '__main__':
    generate_results()

