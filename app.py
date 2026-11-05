import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS AND PROJECT CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "air_quality_output")

# Load the final tuned model and the feature list saved with it.
MODEL_PATH = os.path.join(OUTPUT_DIR, "tuned_extra_trees_model.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "tuned_feature_names.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "tuned_model_metrics.json")
FEATURE_METADATA_PATH = os.path.join(OUTPUT_DIR, "feature_metadata.json")

DEFAULT_VALUES = {
    "PT08.S1(CO)": 1000.0,
    "C6H6(GT)": 5.0,
    "NOx(GT)": 100.0,
    "NO2(GT)": 80.0,
    "T": 20.0,
    "RH": 50.0,
    "AH": 1.0,
}

FEATURE_HELP = {
    "PT08.S1(CO)": "Tin oxide sensor response associated with carbon monoxide.",
    "C6H6(GT)": "Ground-truth benzene concentration.",
    "NOx(GT)": "Ground-truth nitrogen oxides concentration.",
    "NO2(GT)": "Ground-truth nitrogen dioxide concentration.",
    "T": "Ambient temperature (°C).",
    "RH": "Relative humidity (%).",
    "AH": "Absolute humidity.",
}


# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(13, 148, 136, 0.13), transparent 28%),
            #0b1220;
        color: #e5edf7;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.75);
    }

    [data-testid="stSidebar"] {
        background: #101b2d;
        border-right: 1px solid #26364d;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem 2.2rem;
        border: 1px solid rgba(153, 246, 228, 0.18);
        border-radius: 22px;
        background: linear-gradient(120deg, #123c69 0%, #0f766e 100%);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.20);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 2.8rem);
        margin: 0 0 0.5rem 0;
        font-weight: 750;
    }

    .hero p {
        color: #d9f5f1;
        font-size: 1.05rem;
        margin: 0;
    }

    .section-label {
        color: #5eead4;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .info-card {
        background: #142238;
        border: 1px solid #293d57;
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        min-height: 120px;
        margin-bottom: 0.8rem;
    }

    .info-card h3 {
        color: #a9bacd;
        font-size: 0.9rem;
        margin: 0 0 0.55rem 0;
        font-weight: 600;
    }

    .info-card .value {
        color: #5eead4;
        font-size: 1.55rem;
        font-weight: 750;
        margin: 0;
    }

    .result-card {
        background: linear-gradient(135deg, #123c69, #0f766e);
        border: 1px solid rgba(153, 246, 228, 0.25);
        border-radius: 18px;
        padding: 1.6rem;
        margin-top: 1rem;
        text-align: center;
    }

    .result-card .result-label {
        color: #d1fae5;
        font-size: 1rem;
        margin-bottom: 0.45rem;
    }

    .result-card .result-value {
        color: #ffffff;
        font-size: 2.7rem;
        line-height: 1.2;
        font-weight: 800;
    }

    .muted {
        color: #a9bacd;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 11px;
        font-weight: 700;
        border: 1px solid #5eead4;
    }

    div[data-testid="stMetric"] {
        background: #142238;
        border: 1px solid #293d57;
        padding: 1rem;
        border-radius: 14px;
    }

    hr {
        border-color: #293d57;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL AND SUPPORTING FILES
# ============================================================
@st.cache_resource
def load_model_and_features():
    """Load the tuned model and its matching feature names."""
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    # Convert NumPy arrays or other iterable containers to a plain list.
    features = [str(feature) for feature in list(features)]

    if not features:
        raise ValueError("The saved feature list is empty.")

    if len(features) != len(set(features)):
        raise ValueError("The saved feature list contains duplicate names.")

    # Catch an accidental mismatch between model and feature file.
    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and int(expected_count) != len(features):
        raise ValueError(
            f"The model expects {expected_count} features, "
            f"but the feature file contains {len(features)}."
        )

    return model, features


@st.cache_data
def load_json_file(path):
    """Read a JSON file if it exists and contains valid JSON."""
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def show_metric_cards(metrics):
    """Display common evaluation metrics when their names are available."""
    if not isinstance(metrics, dict):
        return

    # Search nested dictionaries too, without assuming one exact JSON schema.
    flattened = {}

    def collect_values(item):
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    flattened[str(key).lower()] = value
                elif isinstance(value, dict):
                    collect_values(value)

    collect_values(metrics)

    metric_aliases = [
        ("R² Score", ("r2", "r2_score", "test_r2", "r2_test")),
        ("MAE", ("mae", "mean_absolute_error", "test_mae")),
        ("RMSE", ("rmse", "root_mean_squared_error", "test_rmse")),
    ]

    available = []
    for label, aliases in metric_aliases:
        value = next((flattened[key] for key in aliases if key in flattened), None)
        if value is not None:
            available.append((label, value))

    if available:
        columns = st.columns(len(available))
        for column, (label, value) in zip(columns, available):
            with column:
                if label == "R² Score":
                    st.metric(label, f"{float(value):.4f}")
                else:
                    st.metric(label, f"{float(value):.4f}")
    else:
        st.info(
            "The metrics file is present, but its keys were not recognised "
            "for summary cards. The complete saved metrics are shown below."
        )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 🌍 Air Quality Intelligence")
st.sidebar.caption("Air Quality Analysis & Prediction using Machine Learning")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict CO Level", "Model Insights", "About Project"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Details")
st.sidebar.write("**Algorithm:** Tuned Extra Trees Regressor")
st.sidebar.write("**Evaluation:** Chronological holdout")
st.sidebar.caption(
    "Performance metrics describe evaluation on the project dataset. "
    "They do not guarantee real-world accuracy."
)


# ============================================================
# CHECK AND LOAD MODEL
# ============================================================
missing_files = [
    path for path in (MODEL_PATH, FEATURES_PATH) if not os.path.exists(path)
]

if missing_files:
    st.error("Required model files were not found.")
    st.write("Please check that these files exist inside `air_quality_output`:")
    for path in missing_files:
        st.code(os.path.basename(path))
    st.stop()

try:
    model, features = load_model_and_features()
except Exception as error:
    st.error(f"The trained model could not be loaded: {error}")
    st.info(
        "Check that `tuned_extra_trees_model.pkl` and "
        "`tuned_feature_names.pkl` were generated by the same notebook run."
    )
    st.stop()


# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">Machine Learning Project</div>
            <h1>Air Quality Intelligence</h1>
            <p>
                Explore air-quality sensor readings and estimate carbon
                monoxide (CO) concentration using a tuned machine learning model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Welcome")
    st.write(
        "This application demonstrates how machine learning can be used "
        "to analyse air-quality measurements and estimate the CO(GT) target "
        "from the sensor and environmental features used during model training."
    )

    first, second, third = st.columns(3)
    with first:
        st.markdown(
            """
            <div class="info-card">
                <h3>Selected Algorithm</h3>
                <p class="value">Extra Trees</p>
                <p class="muted">Hyperparameter-tuned regressor</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with second:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Model Input Features</h3>
                <p class="value">{len(features)}</p>
                <p class="muted">Features loaded from the trained model</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with third:
        st.markdown(
            """
            <div class="info-card">
                <h3>Prediction Target</h3>
                <p class="value">CO(GT)</p>
                <p class="muted">Carbon monoxide concentration</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### What can you do here?")
    left, right = st.columns(2)
    with left:
        st.markdown("**Make a prediction**")
        st.write(
            "Enter the sensor and environmental readings to obtain a "
            "model-based estimate of CO(GT)."
        )
    with right:
        st.markdown("**Explore model information**")
        st.write(
            "Review the saved evaluation metrics and learn about the "
            "dataset and project workflow."
        )

    st.info(
        "Use the **Predict CO Level** page in the sidebar to try the model."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================
elif page == "Predict CO Level":
    st.markdown("## CO Level Prediction")
    st.write(
        "Enter values for the features below. The application uses the exact "
        "feature names and order saved with the tuned model."
    )

    input_values = {}
    columns = st.columns(2)

    for index, feature in enumerate(features):
        with columns[index % 2]:
            input_values[feature] = st.number_input(
                label=feature,
                value=float(DEFAULT_VALUES.get(feature, 0.0)),
                help=FEATURE_HELP.get(
                    feature,
                    "Enter a numeric value for this model feature.",
                ),
                format="%.4f",
                key=f"prediction_input_{feature}",
            )

    st.markdown("---")

    if st.button("Predict CO Level", type="primary"):
        try:
            input_df = pd.DataFrame(
                [[input_values[feature] for feature in features]],
                columns=features,
            )

            if input_df.isnull().any().any():
                st.error("Please provide a value for every input feature.")
                st.stop()

            if not np.isfinite(input_df.to_numpy(dtype=float)).all():
                st.error("Please enter finite numeric values.")
                st.stop()

            prediction = float(model.predict(input_df)[0])

            if not np.isfinite(prediction):
                raise ValueError("The model returned a non-finite prediction.")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Estimated Carbon Monoxide (CO) — CO(GT)
                    </div>
                    <div class="result-value">{prediction:.3f}</div>
                    <div class="result-label">mg/m³</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "The value is a model-generated estimate based on the "
                "entered readings and the project's training data."
            )
            st.warning(
                "This application is an academic machine learning project. "
                "Its output is not an official air-quality measurement, "
                "medical recommendation, or health-safety assessment."
            )

            with st.expander("View submitted input values"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")


# ============================================================
# MODEL INSIGHTS PAGE
# ============================================================
elif page == "Model Insights":
    st.markdown("## Model Insights")
    st.write(
        "The application uses the tuned Extra Trees model saved by the "
        "project notebook."
    )

    metrics = None
    try:
        metrics = load_json_file(METRICS_PATH)
    except Exception as error:
        st.warning(f"Could not read the tuned metrics file: {error}")

    if metrics is not None:
        st.markdown("### Saved evaluation metrics")
        show_metric_cards(metrics)

        with st.expander("View complete metrics data"):
            st.json(metrics)
    else:
        st.info(
            "`tuned_model_metrics.json` was not found. The prediction model "
            "can still work, but the saved evaluation metrics cannot be displayed."
        )

    metadata = None
    try:
        metadata = load_json_file(FEATURE_METADATA_PATH)
    except Exception as error:
        st.warning(f"Could not read feature metadata: {error}")

    st.markdown("### Model input features")
    st.write("The following features are loaded from `tuned_feature_names.pkl`:")
    st.write(", ".join(features))

    if metadata is not None:
        with st.expander("View feature metadata"):
            st.json(metadata)

    st.caption(
        "Evaluation scores should be interpreted in the context of the "
        "chronological train-test split and the dataset used in this project."
    )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================
elif page == "About Project":
    st.markdown("## About the Project")

    st.write(
        "**Air Quality Analysis and Prediction Using Machine Learning** "
        "explores air-quality measurements and builds a regression model "
        "to estimate the CO(GT) target."
    )

    st.markdown("### Project workflow")
    st.markdown(
        """
        1. **Data loading:** Load the Air Quality dataset.
        2. **Data preprocessing:** Handle invalid/missing readings and prepare
           the features for modelling.
        3. **Exploratory data analysis:** Study distributions, relationships
           and patterns in the measurements.
        4. **Model development:** Train and compare regression algorithms.
        5. **Model evaluation:** Evaluate models using a chronological split
           and time-series cross-validation.
        6. **Hyperparameter tuning:** Tune the selected tree-based model.
        7. **Prediction application:** Use the saved tuned model through
           this Streamlit interface.
        """
    )

    st.markdown("### Technology stack")
    st.write(
        "Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · "
        "Joblib · Streamlit"
    )

    st.markdown("### Dataset")
    st.write(
        "The project uses the UCI Air Quality dataset. The model estimates "
        "CO(GT) from the feature columns saved with the trained model."
    )

    st.info(
        "The model estimates CO(GT) from the supplied feature values. "
        "Unless future-time or lagged inputs are explicitly used, this should "
        "be understood as model-based estimation rather than a forecast of "
        "future air quality."
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Air Quality Analysis & Prediction | Machine Learning Project"
)
