from dash_utils.models import get_test_cv_fair_split
from dash_utils.filter import pick_xaxis_color_plt1

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve

# --------------- KiTE Imports ---------------
from KiTE.calibration_models import EWF_calibration
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

sns.set()


# Maybe add 2ndary_cat var that can block by
def plot_id_bias(df=[], trust_features=[], target=None):
    if len(df) <= 0 or len(trust_features) <= 0 or target is None:
        return

    sns.set(style="ticks")
    # Of trust_features, pick x-axis and color
    xlabel, color_col = pick_xaxis_color_plt1(trust_features)

    if xlabel and color_col:
        xlabel_col_indx = trust_features.index(xlabel)
        color_col_indx = trust_features.index(color_col)

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
        ) = get_test_cv_fair_split(df, trust_features, target)

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

            ax.set_ylim([-1, 1])
            # plt.yticks([-0.1, -0.05, 0, 0.05, 0.1], ["-0.1", "-0.05", "0", "0.05", "0.1"])
            # ax.set_xlim([xmin, xmax])

            ax.set_ylabel("Prediction Bias", size=20)
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
            st.warning(f"ERR: Plot could not be generated correctly. Make sure your xaxis is numerical and color_axis is discrete.")
            st.warning(f"Here is the exact error: {e}")
        return plt


def plot_hist_bias(elce2_est, proba, elce_df=[]):
    if len(elce_df) > 0:
        sns.set()
        plt.figure(figsize=(15, 6))

        ax = sns.histplot(elce_df, x="ELCE2", bins=20, kde=True, label="Null ELCE2 Distribution")
        plt.axvline(x=elce2_est, label=f"ELCE2 Estimate (pval = {proba})", color='coral', marker='o', linestyle='dashed',linewidth=5)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={"size": 17})

        plt.title("ELCE2 Distribution for trust_features", size=30)
        plt.ylabel("Frequency", size=25)
        plt.xlabel("ELCE2", size=25)
        return plt


def plot_calibration_curve(y_true, y_pred):
    plt.figure(figsize=(15, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
    ax = sns.lineplot(x=prob_pred, y=prob_true, marker='o', label="Current Calibration", linewidth=3)
    xpoints = ypoints = ax.get_xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False, label="Perfect Calibration")
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    plt.title("Calibration Curve", size=30)
    plt.ylabel("Fraction of positives (Positive Class: 1)", size=20)
    plt.xlabel("Mean Predicted Probability (Positive Class: 1)", size=20)
    plt.legend(prop={"size": 17})

    return plt
