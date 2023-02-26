import streamlit as st
from dash_utils.utils import describe
from dash_utils.constants import model_class_col_name, probability_col_name


def pick_variables(df):
    # Label y-label & trust_features
    describe("Choose a y-label")
    ylabel = st.selectbox(
        "", set(df.columns).difference([model_class_col_name, probability_col_name])
    )

    # Choose fair features:
    describe("Choose features on which to evaluate model trustworthiness")
    trust_features = st.multiselect(
        "",
        set(df.columns).difference(
            [model_class_col_name, probability_col_name, ylabel]
        ),
    )

    return ylabel, trust_features


def pick_xaxis_color_plt1(trust_features):
    describe("Choose an x-axis")
    xlabel = st.selectbox("(This should be Numeric Data)", set(trust_features))

    # Choose fair features:
    describe("Choose a color-column")
    color_col = st.selectbox(
        "(This should be Categorical Data)", set(trust_features).difference([xlabel])
    )

    return xlabel, color_col
