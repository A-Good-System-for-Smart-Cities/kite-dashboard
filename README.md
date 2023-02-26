# KiTE-dashboard [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kite-visualization-tool.streamlit.app/)
> This dashboard a user-friendly interface for non-programmers to use [KiTE](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils) -- a tool that validates and calibrates supervised classification models against bias.

> We hope to empower general users to audit models and develop diagnostic plots that help identify and quantify bias in supervised ML models.

* Policy-makers and general users can use this site to generate the following visualizations:
    1. Prediction bias against a set of features (`fair_features`) in the data provided.
    2. A Histogram Distribution of $ELCE^2$ -- a test statistic that quantifies bias in a set of features (`fair_features`) the user specifies

---
## How to use this site?
1. Collect and pre-process your data as a CSV.
    * Make sure your CSV has your features, labels, and probabilities.
    * Your CSV MUST have the following 2 columns:
        * `model_split_class` -- Accepted values are 'cv' or 'test'
            * **What does this mean?** -- `model_split_class` allows us to identify which rows are for cross-validation (when evaluating fairness) and which rows are for testing.
        * `probability` -- Accepted values are decimal values $\in {0,1}$
            * **What does this mean?** -- For rows with a `model_split_class` of `cv`, `probability` represents the output of the model's prediction probability for the validation data. Similarly, rows with a `model_split_class` of `test` has a `probability` that represents the output of the model's prediction probability for the testing feature set.
    * **Need an Example?** -- Please refer to [notebooks/Preprocess_COMPASS.ipynb](https://github.com/A-Good-System-for-Smart-Cities/kite-dashboard/blob/main/notebooks/Preprocess_COMPASS.ipynb) to see how you can pre-process your data into the right format!
2. Upload your cleaned data!
3. Label which columns are your target (y-label) and which set of features you want to use to evaluate fairness.
4. Generate + Download your plots of interest!


---
## How can I submit feedback/issues?
You can submit any feedback, questions, or issues in the [Issues Tab](https://github.com/A-Good-System-for-Smart-Cities/kite-dashboard/issues) of this Repository. One of our team members will promptly respond to help you out!

---
## How can I safely update this site?
1. Fork this Repo
2. Clone the Repo onto your computer
3. Create a branch (git checkout -b new-feature)
4. Make Changes
5. Run necessary quality assurance tools (Formatter, Linter, etc).
6. Test the site on your local machine with `streamlit run app.py`
7. Add your changes (`git commit -am "Commit Message"` or `git add .` followed by `git commit -m "Commit Message"`)
8. Push your changes to the repo (git push origin new-feature)
9. Create a pull request
