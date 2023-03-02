import streamlit as st
from dash_utils.utils import describe
from dash_utils.constants import model_class_col_name, probability_col_name


def pick_variables(df):
    # Label y-label & trust_features
    options = sorted(
        set(df.columns).difference([model_class_col_name, probability_col_name])
    )
    describe("Choose a y-label")
    ylabel = st.selectbox("Positive Label Class should be 1", options)

    # Choose fair features:
    describe("Choose features on which to evaluate model trustworthiness")
    trust_features = st.multiselect(
        "These features should have >=1 numberic column and >=1 categorical column to generate Bias quantification Plot.",
        sorted(
            set(df.columns).difference(
                [model_class_col_name, probability_col_name, ylabel]
            )
        ),
    )

    return ylabel, trust_features


def pick_xaxis_color_plt1(trust_features):
    describe("Choose an x-axis")
    xlabel = st.selectbox("(This should be Numeric Data)", sorted(set(trust_features)))

    # Choose fair features:
    describe("Choose a stratifying variable")
    color_col = st.selectbox(
        "(This should be Categorical Data)",
        sorted(set(trust_features).difference([xlabel])) + ["None"],
    )

    if color_col == "None":
        color_col = None

    return xlabel, color_col
