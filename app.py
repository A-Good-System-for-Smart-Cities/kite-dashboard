import streamlit as st
from dash_utils.utils import describe, display_header, upload_file
from dash_utils.filter import pick_variables
from dash_utils.plots import plot_id_bias, plot_hist_bias
from dash_utils.data_io import convert_img
from dash_utils.models import run_hyp_test


st.set_page_config(page_icon="🤘", page_title="KITE Dashboard", layout="wide")
m = st.markdown(
    """
    <style>
    div.stDownloadButton > button:first-child {
        background-color: #b07a05;
        color: white;
        height: 5em;
        width: 100%;
        border-radius:10px;
        border:3px
        font-size:16px;
        font-weight: bold;
        margin: auto;
        display: block;
    }

    div.stDownloadButton > button:hover {
        background:linear-gradient(to bottom, #b07a05 5%, #f7d499 100%);
        background-color:#b07a05;
    }

    div.stForm {
        .stFormSubmitButton > button:first-child {
            background-color: #b07a05;
            color: white;
            height: 5em;
            width: 100%;
            border-radius:10px;
            border:3px
            font-size:16px;
            font-weight: bold;
            margin: auto;
            display: block;
        }

        .stFormSubmitButton > button:hover {
            background:linear-gradient(to bottom, #b07a05 5%, #f7d499 100%);
            background-color:#b07a05;
        }
    }
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>""",
    unsafe_allow_html=True,
)

if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False
if "labeled_vars" not in st.session_state:
    st.session_state["labeled_vars"] = False


# Header + Description + link to actual pkg
logo = None  # Image.open("logo-kite.jpg")
display_header(logo, "KITE Dashboard")
describe(
    "What is the point of the site ... how allows users to basically do what we want"
)


# Upload Location --> processes the data
df, success = upload_file()
if success and len(df) > 0:
    st.session_state["file_uploaded"] = True

"""---"""

# Choose Variables -- Generates 2 plots -- 2 tabs
if st.session_state["file_uploaded"]:
    cols = st.columns([0.3, 0.7])
    with cols[0]:
        # Label y-label & fair_features
        target, fair_features = None, None
        with st.form("my_form"):
            target, fair_features = pick_variables(df)

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state["labeled_vars"] = True

    with cols[1]:
        if st.session_state["labeled_vars"]:
            tabs = st.tabs(["Generate a Custom Plot", "Generate Histogram"])

            with tabs[0] as tab:
                ewf_plot = plot_id_bias(df, fair_features, target)
                if ewf_plot:
                    st.pyplot(ewf_plot)
                    img = convert_img(ewf_plot)
                    btn = st.download_button(
                        label="Download EWF Plot",
                        data=img,
                        file_name="ewf_plot.png",
                        mime="image/png",
                    )

            with tabs[1] as tab:
                elce_df = run_hyp_test(df, fair_features, target, num_loops=5)
                hist_plot = plot_hist_bias(elce_df)
                if hist_plot:
                    st.pyplot(hist_plot)
                    img = convert_img(hist_plot)
                    btn = st.download_button(
                        label="Download ELCE2 Histogram",
                        data=img,
                        file_name="elce2_hist.png",
                        mime="image/png",
                    )
        else:
            st.warning("Please label target and fairness features")
