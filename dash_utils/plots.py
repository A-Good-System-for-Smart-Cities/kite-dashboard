import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline

# --------------- KiTE Imports ---------------
from KiTE.calibration_models import EWF_calibration
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge
from dash_utils.models import build_model

plt.style.use("tableau-colorblind10")


# def basic_plt():
#     arr = np.random.normal(1, 1, size=100)
#     fig, ax = plt.subplots()
#     ax.hist(arr, bins=20)
#     return plt


def predicting_recidivism(df):
    np.random.seed(1864)

    (
        X_test,
        X_test_fair,
        y_test,
        X_cv,
        X_cv_fair,
        y_cv,
        prob_test,
        prob_cv,
    ) = build_model(df)

    # kernel hyperparameter
    gamma = 0.5

    # error calibration setup
    n_bins = 20
    kmax = 1
    ewf_model = EWF_calibration()

    # #############################################################################
    #                             Plot calibration plots
    # #############################################################################
    def plot_ewf(X_cv, y_cv, prob_cv, X_test, prob_test, ax, color="blue", label=None):
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
        return ax

    plt.figure(figsize=(2 * 7, 6))
    ax = plt.subplot2grid((2, 2), (0, 0))
    ax.plot([20.0, 70.0], [0, 0], "k-", label="reference")

    mask = X_test_fair.T[1] == 0
    mask *= X_test_fair.T[2] == 1
    ax = plot_ewf(
        X_cv_fair,
        y_cv,
        prob_cv,
        X_test_fair[mask],
        prob_test[mask],
        ax,
        color="#377eb8",
        label="Male",
    )

    mask = X_test_fair.T[1] == 1
    mask *= X_test_fair.T[2] == 1
    ax = plot_ewf(
        X_cv_fair,
        y_cv,
        prob_cv,
        X_test_fair[mask],
        prob_test[mask],
        ax,
        color="#f781bf",
        label="Female",
    )

    ax.set_ylim([-0.1, 0.1])
    plt.yticks([-0.1, -0.05, 0, 0.05, 0.1], ["-0.1", "-0.05", "0", "0.05", "0.1"])
    ax.set_xlim([20, 70.0])

    ax.set_title(r"African-American", size=25, color="indianred")
    ax.set_ylabel("Risk Prediction Bias", size=24)
    ax.set_xlabel(r"Age [years]", size=27)

    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=15)
    ax.legend(loc="upper right", prop={"size": 17})

    ax.grid()
    ax = plt.subplot2grid((2, 2), (0, 1))
    ax.plot([20.0, 70.0], [0, 0], "k-", label="reference")

    mask = X_test_fair.T[1] == 0
    mask *= X_test_fair.T[2] == 0
    ax = plot_ewf(
        X_cv_fair,
        y_cv,
        prob_cv,
        X_test_fair[mask],
        prob_test[mask],
        ax,
        color="#377eb8",
        label="Male",
    )

    mask = X_test_fair.T[1] == 1
    mask *= X_test_fair.T[2] == 0
    ax = plot_ewf(
        X_cv_fair,
        y_cv,
        prob_cv,
        X_test_fair[mask],
        prob_test[mask],
        ax,
        color="#f781bf",
        label="Female",
    )

    ax.set_ylim([-0.1, 0.1])
    plt.yticks([-0.1, -0.05, 0, 0.05, 0.1], 5 * [""])
    ax.set_xlim([20, 70.0])

    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=15)

    ax.set_xlabel(r"Age [years]", size=27)
    ax.set_title(r"Non-African-American", size=25, color="indianred")

    ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.02)
    return plt
