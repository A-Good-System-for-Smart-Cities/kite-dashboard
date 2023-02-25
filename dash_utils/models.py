import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dash_utils.constants import model_class_col_name, probability_col_name
from KiTE.metrics import ELCE2

plt.style.use("tableau-colorblind10")


def get_test_cv_fair_split(df, fair_features, target):
    features = set(df.columns).difference(
        [model_class_col_name, probability_col_name, target]
    )

    validate = df[df[model_class_col_name] == "cv"]
    test = df[df[model_class_col_name] == "test"]

    X_cv = np.array(validate[features])
    X_test = np.array(test[features])

    X_cv_fair = np.array(validate[fair_features])
    X_test_fair = np.array(test[fair_features])

    y_cv = np.array(validate[target])
    y_test = np.array(test[target])

    prob_cv = np.array(validate[probability_col_name])  # clf.predict_proba(X_cv)[:, 1]
    prob_test = np.array(test[probability_col_name])  # clf.predict_proba(X_test)[:, 1]

    return X_test, X_test_fair, y_test, X_cv, X_cv_fair, y_cv, prob_test, prob_cv


@st.cache_data
def run_hyp_test(df, fair_features, target, num_loops=100):
    (
        _,
        X_test_fair,
        y_test,
        _2,
        _3,
        _4,
        prob_test,
        prob_cv,
    ) = get_test_cv_fair_split(df, fair_features, target)

    X_test = X_test_fair
    prob_kernel_wdith = 0.1
    gamma = 0.5

    prob_cal = prob_test.copy()
    X_test = X_test[0 <= prob_cal]
    y_test = y_test[0 <= prob_cal]
    prob_cal = prob_cal[0 <= prob_cal]

    X_test = X_test[1 >= prob_cal]
    y_test = y_test[1 >= prob_cal]
    prob_cal = prob_cal[1 >= prob_cal]

    if len(X_test) > 0 and len(y_test) > 0 and len(prob_cal > 0):
        progress_text = "Calculating ELCE2 ... "
        my_bar = st.progress(0, text=progress_text)

        elces2 = []
        for i in range(num_loops):
            ELCE2_, nulls, pvalue = ELCE2(
                X_test,
                y_test,
                prob_cal,
                prob_kernel_width=prob_kernel_wdith,
                kernel_function="rbf",
                gamma=gamma,
                random_state=np.random.RandomState(1864),
                iterations=200,
                verbose=False,
            )

            if pvalue > 0.49:
                pvalue = 0.49

            if ELCE2_ < 0.0:
                ELCE2_ = -0.00005
            for n in nulls:
                elces2.append({"ELCE2": n, "pval": pvalue})
            my_bar.progress((i + 1) / num_loops, text=progress_text)

        elce_df = pd.DataFrame(elces2)

        return elce_df
