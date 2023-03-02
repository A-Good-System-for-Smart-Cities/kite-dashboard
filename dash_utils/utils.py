import streamlit as st
import pandas as pd
import numpy as np
from dash_utils.data_io import load_input_csv

threshold = 5000


def describe(desc):
    st.markdown(f"##### {desc}")


description = """
This dashboard a user-friendly interface for non-programmers to use [KiTE](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils) -- a tool that validates and calibrates supervised classification models against bias.

> We hope to empower general users to audit models and develop diagnostic plots that help identify and quantify bias in supervised ML models.

* Policy-makers and general users can use this site to generate the following visualizations:
    1. A calibration curve to compare the calibration of the model's probabilistic predictions.
    2. Model Bias quantification curves, where you can plot Prediction Bias against a set of features (`trust_features`) in the data provided.
    3. Model Trustworthiness hypothesis testing curve based on $ELCE^2$ -- a test statistic that quantifies bias in a set of features (`trust_features`) the user specifies
"""

file_reqs = """
    * Make sure your CSV has your features, labels, and probabilities.
    * **Your CSV MUST have the following column**:
        * `probability` -- Accepted values are decimal values $\in [0,1]$
            * **What does this mean?** -- `probability` represents the prediction probability for the feature set.
    * **Need an Example?**
        * Example 1: [Preprocessed BROWARD COMPASS Data](https://github.com/A-Good-System-for-Smart-Cities/kite-dashboard/blob/main/sample_data/compass.csv)
        * Example 2: [Homeownership Data](https://github.com/A-Good-System-for-Smart-Cities/kite-dashboard/blob/main/sample_data/home_ownership.csv)

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
    np.random.seed(1864)
    st.markdown(f"""
    # Upload a Input File.
    > NOTE: To limit our dashboard's carbon footprint, we limit the number of input observations to {threshold}.
    """)
    with st.expander("If you want to run our tool on larger input sizes, please contact our development team to audit these larger data sizes. Here is our contact information:"):
        st.markdown("""
        * **Option 1**: Submit any feedback, questions, or issues in the [Issues Tab](https://github.com/A-Good-System-for-Smart-Cities/kite-dashboard/issues) of this Repository. One of our team members will promptly respond to help you out!
        * **Option 2**: Email us at arya.farahi@austin.utexas.edu for any additional questions.
        """)

    st.markdown(file_reqs)
    uploaded_file = st.file_uploader("Please upload a CSV", type="csv")
    df = pd.DataFrame()
    success = False
    if uploaded_file is not None:
        df, success = load_input_csv(uploaded_file)
        if len(df) >= threshold:
            st.warning(
                f"Your file had {len(df)} rows. We will only process a random sample of {threshold} rows."
            )
            df = df.sample(n=threshold)
    else:
        st.warning("Valid file not yet uploaded")
    return df, success
