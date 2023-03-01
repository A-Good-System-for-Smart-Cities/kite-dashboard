import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dash_utils.constants import probability_col_name
from KiTE.metrics import ELCE2

plt.style.use("tableau-colorblind10")


@st.cache_data
def get_test_cv_fair_split(
    df, trust_features, encoding_req_names, target, ret_trust_features=False
):
    np.random.seed(1864)

    feature_and_proba = list(set(df.columns).difference([target]))

    df = df.copy()
    if encoding_req_names:
        for col in encoding_req_names:
            # Get one hot encoding of columns B
            if col in feature_and_proba:
                one_hot = pd.get_dummies(df[col])
                # Drop column B as it is now encoded
                df = df.drop(col, axis=1)
                # Join the encoded df
                feature_and_proba.remove(col)
                feature_and_proba = [*feature_and_proba, *one_hot.columns]

                if col in trust_features:
                    trust_features.remove(col)
                    trust_features = [*trust_features, *one_hot.columns]

                df = pd.concat([df, one_hot], axis=1)
            if df.memory_usage(index=True, deep=True).sum() > 1000000:
                st.warning(
                    f"## Current memory usage ({df.memory_usage(index=True, deep=True).sum() // (1**7)} MB) EXCEEDS Threshold (1MB)"
                )
                return

    st.markdown(
        f"> Current memory usage: {df.memory_usage(index=True, deep=True).sum()} bytes"
    )
    if df.memory_usage(index=True, deep=True).sum() > 1000000:
        st.warning(
            f"> Current memory usage ({df.memory_usage(index=True, deep=True).sum() // (1**7)} MB) EXCEEDS Threshold (1MB)"
        )
        return

    features = list(set(feature_and_proba).difference([probability_col_name, target]))

    X_cv_and_proba, X_test_and_proba, y_cv, y_test = train_test_split(
        df[feature_and_proba], df[target], test_size=0.5, random_state=42
    )
    validate = pd.concat([X_cv_and_proba, y_cv], axis=1)
    test = pd.concat([X_test_and_proba, y_test], axis=1)

    X_cv = np.array(validate[features])
    X_test = np.array(test[features])

    X_cv_fair = np.array(validate[trust_features])
    X_test_fair = np.array(test[trust_features])

    y_cv = np.array(validate[target])
    y_test = np.array(test[target])

    prob_cv = np.array(validate[probability_col_name])  # clf.predict_proba(X_cv)[:, 1]
    prob_test = np.array(test[probability_col_name])  # clf.predict_proba(X_test)[:, 1]

    if ret_trust_features:
        return (
            X_test,
            X_test_fair,
            y_test,
            X_cv,
            X_cv_fair,
            y_cv,
            prob_test,
            prob_cv,
            trust_features,
            df,
        )

    return X_test, X_test_fair, y_test, X_cv, X_cv_fair, y_cv, prob_test, prob_cv


@st.cache_data
def run_hyp_test(df, trust_features, encoding_req_names, target=None, num_loops=5):
    if len(df) <= 0 or len(trust_features) <= 0 or target is None:
        return None

    output = get_test_cv_fair_split(df, trust_features, encoding_req_names, target)

    if output is None:
        err = "One-Hot Encoding made this file too big! Reduce the number of rows."
        return None, err

    (
        X_test,
        X_test_fair,
        y_test,
        _2,
        _3,
        _4,
        prob_test,
        _5,
    ) = output

    if X_test is None:
        err = "One-Hot Encoding made this file too big! Reduce the number of rows."
        return None, err

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

    elce2_est = 0
    proba = 1

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

            elce2_est = ELCE2_
            proba = pvalue
            for n in nulls:
                elces2.append({"ELCE2": n, "pval": pvalue})
            my_bar.progress((i + 1) / num_loops, text=progress_text)

        elce_df = pd.DataFrame(elces2)

        return elce2_est, proba, elce_df
