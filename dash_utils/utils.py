import streamlit as st

# import seaborn as sns
# from PIL import Image
# import base64
# from pathlib import Path
import pandas as pd

from dash_utils.data_io import load_full_csv


def describe(desc):
    st.markdown(f"##### {desc}")


################################################################################
#                                   HEADER
################################################################################
def display_header(logo, page_title):
    col0, col1 = st.columns([0.7, 0.3])
    with col0:
        st.markdown(
            """<style> .font {font-size:45px ; font-family: 'Cooper Black'; color: #FF9633;} </style>""",
            unsafe_allow_html=True,
        )
        st.markdown(f'<p class="font">{page_title}</p>', unsafe_allow_html=True)
    with col1:
        if logo:
            st.image(logo)
    st.subheader("Put description + usage here")


################################################################################
#                                   UPLOAD
################################################################################
def upload_file():
    st.header("Upload a Input File")
    describe("What kind of input file ... how needs feature cols, and label col")
    uploaded_file = st.file_uploader("")
    df = pd.DataFrame()
    success = False
    if uploaded_file is not None:
        df, success = load_input_csv(uploaded_file)
    else:
        st.warning("Valid file not yet uploaded")
    return df, success
