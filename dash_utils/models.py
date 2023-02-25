import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from dash_utils.constants import model_class_col_name, probability_col_name


plt.style.use("tableau-colorblind10")


def get_test_cv_fair_split(df, fair_features, target):
    features = set(df.columns).difference([model_class_col_name, probability_col_name, target])

    validate = df[df[model_class_col_name] == 'cv']
    test = df[df[model_class_col_name] == 'test']

    X_cv = np.array(validate[features])
    X_test = np.array(test[features])

    X_cv_fair = np.array(validate[fair_features])
    X_test_fair = np.array(test[fair_features])

    y_cv = np.array(validate[target])
    y_test = np.array(test[target])

    prob_cv = np.array(validate[probability_col_name])  # clf.predict_proba(X_cv)[:, 1]
    prob_test = np.array(test[probability_col_name])    # clf.predict_proba(X_test)[:, 1]

    return X_test, X_test_fair, y_test, X_cv, X_cv_fair, y_cv, prob_test, prob_cv


def run_hyp_test():
    #
    pass
