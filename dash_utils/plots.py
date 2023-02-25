import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.pipeline import make_pipeline

# --------------- KiTE Imports ---------------
from KiTE.calibration_models import EWF_calibration
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge
from dash_utils.models import get_test_cv_fair_split
from dash_utils.filter import pick_xaxis_color_plt1

plt.style.use("tableau-colorblind10")


def predicting_recidivism(df, fair_features, target):
    # Of fair_features, pick x-axis and color
    xlabel, color_col = pick_xaxis_color_plt1(fair_features)

    if xlabel and color_col:
        color_col_indx = fair_features.index(color_col)
        st.write(color_col_indx)
        xmin = np.min(df[xlabel])
        xmax = np.max(df[xlabel])
        st.write(xmin, xmax)
        colors_vals = pd.Series(df[color_col]).unique()
        colors = sns.husl_palette(len(colors_vals))

        np.random.seed(1864)
        gamma = 0.5     # kernel hyperparameter
        n_bins = 20     # error calibration setup
        kmax = 1
        ewf_model = EWF_calibration()

        X_test, X_test_fair, y_test, X_cv, X_cv_fair, y_cv, prob_test, prob_cv = get_test_cv_fair_split(df, fair_features, target)

        # #############################################################################
        #                             Plot calibration plots
        # #############################################################################
        def plot_ewf(X_cv, y_cv, prob_cv, X_test, prob_test, ax, color="blue", label=None):
            st.write("plotting swf")
            # Train a calibration method (EWF) on 2nd data subset
            ewf_model.fit(X_cv, prob_cv, y_cv, kernel_function="rbf", gamma=gamma)
            ewf = ewf_model.predict(X_test, prob_test, mode="bias")

            # Plot outcome
            ax.plot(
                X_test.T[0] * 10 + 0.3 * (np.random.random() - 0.5), ewf, ".", color=color
            )

            Xfit = X_test.T[0] * 10
            model = make_pipeline(PolynomialFeatures(3), Ridge())
            model.fit(Xfit[:, np.newaxis], ewf)

            X_plot = np.linspace(20, 70, 201)
            y_plot = model.predict(X_plot[:, np.newaxis])

            ax.plot(X_plot, y_plot, lw=5, color=color, label=label)
            st.write("Done")
            return ax

        plt.figure(figsize=(2 * 7, 6))
        ax = plt.subplot2grid((2, 2), (0, 0))

        # ax.plot([xmin, xmax], [0, 0], "k-", label="reference")

        for c, clr in enumerate(colors):
            label = colors_vals[c]
            mask = X_test_fair.T[color_col_indx] == label
            ax = plot_ewf(
                X_cv_fair,
                y_cv,
                prob_cv,
                X_test_fair[mask],
                prob_test[mask],
                ax,
                color=clr,
                label=label
            )

        # ax.set_ylim([-0.1, 0.1])
        # plt.yticks([-0.1, -0.05, 0, 0.05, 0.1], ["-0.1", "-0.05", "0", "0.05", "0.1"])
        # ax.set_xlim([xmin, xmax])

        ax.set_ylabel("Risk Prediction Bias", size=24)
        ax.set_xlabel(xlabel, size=27)

        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.tick_params(axis="both", which="minor", labelsize=15)
        ax.legend(prop={"size": 17})

        ax.grid()
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.02)
        return plt
