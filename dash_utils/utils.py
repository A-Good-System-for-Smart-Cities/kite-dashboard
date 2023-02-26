import streamlit as st
import pandas as pd
from dash_utils.data_io import load_input_csv


def describe(desc):
    st.markdown(f"##### {desc}")



description = """
This dashboard a user-friendly interface for non-programmers to use [KiTE](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils) -- a tool that validates and calibrates supervised classification models against bias.

> We hope to empower general users to audit models and develop diagnostic plots that help identify and quantify bias in supervised ML models.

* Policy-makers and general users can use this site to generate the following visualizations:
    1. Prediction bias against a set of features (`trust_features`) in the data provided.
    2. A Histogram Distribution of $ELCE^2$ -- a test statistic that quantifies bias in a set of features (`trust_features`) the user specifies

"""

file_reqs = """
    * Make sure your CSV has your features, labels, and probabilities.
    * **Your CSV MUST have the following column**:
        * `probability` -- Accepted values are decimal values $\in {0,1}$
            * **What does this mean?** -- For rows with a `model_split_class` of `cv`, `probability` represents the output of the model's prediction probability for the validation data. Similarly, rows with a `model_split_class` of `test` has a `probability` that represents the output of the model's prediction probability for the testing feature set.
    * **Need an Example?** -- Please refer to [notebooks/Preprocess_COMPASS.ipynb](https://github.com/A-Good-System-for-Smart-Cities/kite-dashboard/blob/main/notebooks/Preprocess_COMPASS.ipynb) to see how you can pre-process your data into the right format!

"""

how_to_use = f"""
1. Collect and pre-process your data as a CSV.
{file_reqs}
2. Upload your cleaned data!
3. Label which columns are your target (y-label) and which set of features you want to use to evaluate trustworthiness.
4. Generate + Download your plots of interest!

"""


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
        st.markdown(description)
    with col1:
        if logo:
            st.image(logo)

    st.markdown(
        """
    ---
    ### How to use this site?
    """
    )
    with st.expander("Click to see detailed instructions."):
        st.markdown(how_to_use)

    st.markdown("---")


################################################################################
#                                   UPLOAD
################################################################################
def upload_file():
    st.header("Upload a Input File")
    st.markdown(file_reqs)
    uploaded_file = st.file_uploader("")
    df = pd.DataFrame()
    success = False
    if uploaded_file is not None:
        df, success = load_input_csv(uploaded_file)
    else:
        st.warning("Valid file not yet uploaded")
    return df, success
