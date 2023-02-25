import streamlit as st

from dash_utils.utils import describe, display_header, upload_file

# from dash_utils.filter import pick_variables
from dash_utils.plots import basic_plt, predicting_recidivism
import matplotlib.pyplot as plt

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

# Choose Variables -- Generates 2 plots -- 2 tabs
if success and len(df) > 0:
    tabs = st.tabs(["Generate a Custom Plot", "Generate Histogram"])
    with tabs[0] as tab:
        "Put custom plot"
        st.pyplot(predicting_recidivism(df))
        # X_cv, y_cv, prob_cv, X_test, prob_test, xlabel, ylabel = pick_variables(df)
    with tabs[1] as tab:
        "put hist"
else:
    st.warning("Something went wrong with the uploaded file.")
