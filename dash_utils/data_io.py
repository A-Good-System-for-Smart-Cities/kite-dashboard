import streamlit as st
import pandas as pd
from dash_utils.constants import model_class_col_name, probability_col_name
import io


def load_full_csv(path: str):
    df = None
    success = False
    try:
        df = pd.read_csv(path)
        success = True
    except Exception as e:
        st.warning(f"This data {path} is unavailable. {e}")
    return df, success


def load_input_csv(path: str):
    df, success = load_full_csv(path)

    if model_class_col_name not in df.columns or probability_col_name not in df.columns:
        st.warning(
            f"Either '{model_class_col_name}' or '{probability_col_name}' NOT in CSV columns."
        )
        success = False
    return df, success


@st.cache_data
def convert_df(df, mode="csv"):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    if mode == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif mode == "pickle":
        return df.to_pickle()
    return None


@st.cache_data
def convert_img(plt):
    img = io.BytesIO()
    plt.savefig(img, format="png")

    return img
