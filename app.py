import streamlit as st
import matplotlib.pyplot as plt

from dash_utils.utils import describe, display_header, upload_file
from dash_utils.filter import pick_variables
from dash_utils.plots import predicting_recidivism
from dash_utils.models import get_test_cv_fair_split


plt.style.use("tableau-colorblind10")


# Header + Description + link to actual pkg
st.set_page_config(page_icon="🤘", page_title="KITE Dashboard", layout="wide")
logo = None  # Image.open("logo-kite.jpg")
display_header(logo, "KITE Dashboard")
describe(
    "What is the point of the site ... how allows users to basically do what we want"
)

# Upload Location --> processes the data
df, success = upload_file()

"""---"""

# Choose Variables -- Generates 2 plots -- 2 tabs
if success and len(df) > 0:
    cols = st.columns([0.3, 0.7])
    with cols[0]:
        # Label y-label & fair_features
        target, fair_features = pick_variables(df)
    with cols[1]:
        tabs = st.tabs(["Generate a Custom Plot", "Generate Histogram"])

        with tabs[0] as tab:
            ewf_plot = predicting_recidivism(df, fair_features, target)
            if ewf_plot:
                st.pyplot(ewf_plot)
        with tabs[1] as tab:
            "put hist"
