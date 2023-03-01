import streamlit as st
from PIL import Image
from dash_utils.utils import display_header, upload_file
from dash_utils.filter import pick_variables
from dash_utils.plots import plot_id_bias, plot_hist_bias, plot_calibration_curve
from dash_utils.data_io import convert_img
from dash_utils.models import run_hyp_test
from dash_utils.constants import probability_col_name


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
target_colname, trust_features_names, encoding_req_names = None, [], []

# Choose Variables -- Generates 2 plots -- 2 tabs
if st.session_state["file_uploaded"]:
    cols = st.columns([0.3, 0.7])
    with cols[0]:
        # Label y-label & trust_features
        target_colname, trust_features_names, encoding_req_names = pick_variables(df)
        if len(encoding_req_names) == 0:
            encoding_req_names = None

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
            or target_colname is None
            or trust_features_names is None
            or len(trust_features_names) <= 0
        ):
            st.warning("Please label target and trustworthiness features")
        else:
            tabs = st.tabs(
                [
                    "Generate Calibration Curve",
                    "Generate a Custom Plot",
                    "Generate Histogram",
                ]
            )
            with tabs[0] as tab:
                st.markdown(
                    """
                * This [Calibration Curve/Reliability Diagram](https://scikit-learn.org/stable/modules/calibration.html) compares the calibration of the model's probabilistic predictions.
                * To generate this plot, we calculate the Mean Predicted Probability and the Fraction of positives from your input data's target/y-label and probability columns.
                """
                )
                y_test = df[target_colname]
                prob_test = df[probability_col_name]

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

            with tabs[1] as tab:
                # df=[], trust_features=[], encoding_req_names = [], target=None
                output = plot_id_bias(
                    df=df,
                    trust_features=trust_features_names,
                    encoding_req_names=[],
                    target=target_colname,
                )
                if output and len(output) == 1:
                    ewf_plot = output[0]
                    st.markdown(
                        """
                    * This plot helps identify regions of potential bias in the given dataset.
                    * We calculate prediction bias using the Error Witness Function (EWF) -- a metric that calcualtes the discrepancy between observed labels and predicted probabilities.
                    * To generate this plot, we split the data randomly into 50% validation and 50% testing. We train an EWF model on the validation data, and use it to predict probabilities on the testing set.
                    * We stratify data on the categorical variable chosen.
                    """
                    )
                    st.pyplot(ewf_plot)
                    img = convert_img(ewf_plot)
                    btn = st.download_button(
                        label="Download EWF Plot",
                        data=img,
                        file_name="ewf_plot.png",
                        mime="image/png",
                    )
                elif output and len(output) == 2:
                    err = output[1]
                    st.warning(f"Plot could not be generated because {err}")

            with tabs[2] as tab:
                output = run_hyp_test(
                    df, trust_features_names, encoding_req_names, target_colname
                )
                if output and len(output) == 3:
                    elce2_est, proba, elce_df = output
                    hist_plot = plot_hist_bias(elce2_est, proba, elce_df)
                    if hist_plot:
                        st.markdown(
                            """
                        * This plot helps quantify local bias (using the $ELCE^2$ statistic) based on the features used to evaluate trustworthiness of the given dataset.
                        * We run a bootstrapped ELCE2 calculation that generates a Null Distribution, $ELCE^2$ estimate, and probability 5 times.
                        * The $ELCE^2$ estimate (orange line) represents local bias when compared to the null distribution centered at 0.
                            * If the pvalue (aka probability) < alpha = 0.05, then we can reject the Null Hypothesis. In that case, we have convincing statistical evidence that the model is locally biased on the trustworthiness features.
                            * Otherwise, if pvalue >= alpha = 0.05, then we fail to reject the Null Hypothesis as we lack convincing statistical evidence that the model is locally biased on our chosen set of trust_features
                        """
                        )
                        st.pyplot(hist_plot)
                        img = convert_img(hist_plot)
                        btn = st.download_button(
                            label="Download ELCE2 Histogram",
                            data=img,
                            file_name="elce2_hist.png",
                            mime="image/png",
                        )
                    else:
                        st.warning("Plot could not be generated.")
                elif output and len(output) == 2:
                    err = output[1]
                    st.warning(f"Plot could not be generated because {err}")
