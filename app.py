import streamlit as st
from PIL import Image
from dash_utils.utils import display_header, upload_file
from dash_utils.filter import pick_variables
from dash_utils.plots import plot_id_bias, plot_hist_bias, plot_calibration_curve
from dash_utils.data_io import convert_img
from dash_utils.models import run_hyp_test, get_test_cv_fair_split


st.set_page_config(page_icon="🤘", page_title="KITE Dashboard", layout="wide")
m = st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6{
        color: #FF9633;
        font-family: 'Cooper Black';
        font-variant: small-caps;
        text-transform: none;
        font-weight: 200;
        margin-bottom: 0px;
    }

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
logo = Image.open("logo-kite.jpg")
display_header(logo, "KITE Dashboard")

# Upload Location --> processes the data
df, success = upload_file()
if success and len(df) > 0:
    st.session_state["file_uploaded"] = True

"""---"""
target, trust_features = None, None
# Choose Variables -- Generates 2 plots -- 2 tabs
if st.session_state["file_uploaded"]:
    cols = st.columns([0.3, 0.7])
    with cols[0]:
        # Label y-label & trust_features
        target, trust_features = pick_variables(df)
        with st.form("my_form"):
            # Every form must have a submit button.
            submitted = st.form_submit_button(
                "Generate Plots!", use_container_width=True
            )
            if submitted:
                st.session_state["labeled_vars"] = True

    with cols[1]:
        if (
            not st.session_state["labeled_vars"]
            or target is None
            or trust_features is None
            or len(trust_features) <= 0
        ):
            st.warning("Please label target and trustworthiness features")
        else:
            tabs = st.tabs(
                [
                    "Generate a Custom Plot",
                    "Generate Histogram",
                    "Generate Calibration Curve",
                ]
            )

            with tabs[0] as tab:
                ewf_plot = plot_id_bias(df, trust_features, target)
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
                elce2_est, proba, elce_df = run_hyp_test(
                    df, trust_features, target, num_loops=5
                )
                hist_plot = plot_hist_bias(elce2_est, proba, elce_df)
                if hist_plot:
                    st.pyplot(hist_plot)
                    img = convert_img(hist_plot)
                    btn = st.download_button(
                        label="Download ELCE2 Histogram",
                        data=img,
                        file_name="elce2_hist.png",
                        mime="image/png",
                    )
            with tabs[2] as tab:
                (
                    _,
                    _2,
                    y_test,
                    _3,
                    _4,
                    _5,
                    prob_test,
                    _6,
                ) = get_test_cv_fair_split(df, trust_features, target)

                calib_plt = plot_calibration_curve(y_test, prob_test)
                if calib_plt:
                    st.pyplot(calib_plt)
                    img = convert_img(calib_plt)
                    btn = st.download_button(
                        label="Download Calibration Curve",
                        data=img,
                        file_name="calib_plot.png",
                        mime="image/png",
                    )
