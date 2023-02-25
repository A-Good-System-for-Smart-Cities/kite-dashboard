from dash_utils.models import get_test_cv_fair_split
from dash_utils.filter import pick_xaxis_color_plt1

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.pipeline import make_pipeline

# --------------- KiTE Imports ---------------
from KiTE.calibration_models import EWF_calibration
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

sns.set()


# Maybe add 2ndary_cat var that can block by
def plot_id_bias(df, fair_features, target):
    sns.set(style="ticks")
    # Of fair_features, pick x-axis and color
    xlabel, color_col = pick_xaxis_color_plt1(fair_features)

    if xlabel and color_col:
        xlabel_col_indx = fair_features.index(xlabel)
        color_col_indx = fair_features.index(color_col)

        xmin = np.min(df[xlabel])
        xmax = np.max(df[xlabel])

        colors_vals = pd.Series(df[color_col]).unique()
        colors = sns.husl_palette(len(colors_vals))
        dark_colors = sns.husl_palette(len(colors_vals), l=0.4)

        np.random.seed(1864)
        gamma = 0.5  # kernel hyperparameter
        ewf_model = EWF_calibration()

        (
            X_test,
            X_test_fair,
            y_test,
            X_cv,
            X_cv_fair,
            y_cv,
            prob_test,
            prob_cv,
        ) = get_test_cv_fair_split(df, fair_features, target)

        # #############################################################################
        #                             Plot calibration plots
        # #############################################################################
        def plot_ewf(
            X_cv, y_cv, prob_cv, X_test, prob_test, ax, color_indx=0, label=None
        ):
            # Train a calibration method (EWF) on 2nd data subset
            ewf_model.fit(X_cv, prob_cv, y_cv, kernel_function="rbf", gamma=gamma)
            ewf = ewf_model.predict(X_test, prob_test, mode="bias")

            # Plot outcome
            ax.plot(
                X_test.T[xlabel_col_indx], ewf, ".", color=colors[color_indx], alpha=0.5
            )

            Xfit = X_test.T[xlabel_col_indx]
            model = make_pipeline(PolynomialFeatures(3), Ridge())
            model.fit(Xfit[:, np.newaxis], ewf)

            X_plot = np.linspace(xmin, xmax, 201)
            y_plot = model.predict(X_plot[:, np.newaxis])

            ax.plot(X_plot, y_plot, lw=3, color=dark_colors[color_indx], label=label)
            return ax

        try:
            plt.figure(figsize=(15, 6))
            ax = plt.subplot2grid((2, 2), (0, 0))

            ax.plot([xmin, xmax], [0, 0], "k-", label="Reference")

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
                    color_indx=c,
                    label=label,
                )

            # ax.set_ylim([-0.1, 0.1])
            # plt.yticks([-0.1, -0.05, 0, 0.05, 0.1], ["-0.1", "-0.05", "0", "0.05", "0.1"])
            # ax.set_xlim([xmin, xmax])

            ax.set_ylabel("Risk Prediction Bias", size=20)
            ax.set_xlabel(xlabel, size=20)

            # ax.tick_params(axis="both", which="major", labelsize=15)
            # ax.tick_params(axis="both", which="minor", labelsize=15)
            ax.legend(prop={"size": 17})
            for tick in ax.get_yticklabels():
                tick.set_visible(True)

            ax.grid()
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.02)
        except Exception as e:
            st.warning(f"Plot could not be generated correctly due to {e}")
        return plt


def plot_hist_bias(elce_df):
    if len(elce_df) > 0:
        plt.figure(figsize=(15, 6))

        sns.histplot(elce_df, x="ELCE2", bins=20, kde=True)
        plt.title("ELCE2 Distribution for fair_features", size=30)

        plt.ylabel("Frequency", size=25)
        plt.xlabel("ELCE2", size=25)
        return plt
