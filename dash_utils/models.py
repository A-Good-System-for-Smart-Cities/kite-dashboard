import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


plt.style.use("tableau-colorblind10")


def build_model(df):
    features = [
        "age",
        "sex",
        "African-American",
        "juv_fel_count",
        "juv_misd_count",
        "priors_count",
        "c_charge_degree",
        "Asian",
        "Caucasian",
        "Hispanic",
        "Native American",
        "Other",
    ]
    fair_features = ["age", "sex", "African-American"]
    target = "is_recid"

    # Split data into train, validate and test data
    train, validate, test = np.split(
        df.sample(frac=1), [int(0.33 * len(df)), int(0.66 * len(df))]
    )

    X_train = np.array(train[features])
    X_cv = np.array(validate[features])
    X_test = np.array(test[features])

    X_cv_fair = np.array(validate[fair_features])
    X_test_fair = np.array(test[fair_features])

    y_train = np.array(train[target])
    y_cv = np.array(validate[target])
    y_test = np.array(test[target])

    # Train the Random Forest model on the 1st subset of data (training set)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    prob_cv = clf.predict_proba(X_cv)[:, 1]

    return X_test, X_test_fair, y_test, X_cv, X_cv_fair, y_cv, prob_test, prob_cv


def run_hyp_test():
    #
    pass
