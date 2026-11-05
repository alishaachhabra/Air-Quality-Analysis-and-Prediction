# Air Quality Analysis and Prediction Using Machine Learning

An end-to-end Machine Learning project that explores air quality measurements, compares regression algorithms, and estimates carbon monoxide (CO) concentration using a tuned Extra Trees Regressor.

The project includes data preprocessing, exploratory data analysis, model evaluation, hyperparameter tuning, and an interactive Streamlit application.

## Project Overview

Air pollution is an important environmental concern. Analysing air quality measurements can help us understand relationships between different pollutants and environmental conditions.

This project uses the UCI Air Quality dataset to analyse sensor readings and develop machine learning regression models for estimating carbon monoxide concentration (CO(GT)).

## Project Objectives

- Perform exploratory data analysis on air quality measurements.
- Clean and preprocess the dataset.
- Analyse relationships between pollutants and environmental variables.
- Train and compare multiple regression algorithms.
- Evaluate models using chronological validation.
- Perform hyperparameter tuning to improve model performance.
- Develop an interactive Streamlit application for predictions.

## Dataset

**Dataset:** UCI Air Quality Dataset

The dataset contains hourly air-quality measurements collected using an array of chemical sensors.

The project uses the following input features:

- PT08.S1(CO)
- C6H6(GT)
- NOx(GT)
- NO2(GT)
- Temperature (T)
- Relative Humidity (RH)
- Absolute Humidity (AH)

**Prediction Target:** CO(GT) – Carbon Monoxide concentration.

The dataset contains `-200` values representing missing measurements, which are handled during preprocessing.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git and GitHub |

## Machine Learning Workflow

1. **Data Collection:** Load the UCI Air Quality dataset.
2. **Data Preprocessing:** Handle missing values and prepare the input features and prediction target.
3. **Exploratory Data Analysis:** Investigate distributions, correlations and relationships between variables.
4. **Model Development:** Train and compare multiple regression algorithms.
5. **Model Evaluation:** Evaluate model performance using chronological train-test splitting and time-series cross-validation.
6. **Hyperparameter Tuning:** Optimise the selected Extra Trees Regressor.
7. **Model Interpretation:** Use feature importance and SHAP analysis to understand model behaviour.
8. **Deployment Interface:** Integrate the trained model into an interactive Streamlit application.

## Models Evaluated

The project experiments with the following regression algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbours Regressor
- Support Vector Regression (SVR)

The final model selected for the application is the **tuned Extra Trees Regressor**.

Model selection considers chronological holdout performance and time-series cross-validation rather than relying on a single random train-test split.

## Final Model Performance

The tuned Extra Trees model achieved the following results on the chronological holdout evaluation:

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.8517 |
| Mean Absolute Error (MAE) | 0.3448 |
| Root Mean Squared Error (RMSE) | 0.5178 |

These results indicate that the model explains approximately 85.17% of the variance in the evaluated target values.

*Note: These metrics represent performance on the project's evaluation dataset. They do not guarantee real-world accuracy.*

## Streamlit Application

The project includes an interactive web application built using Streamlit.

### Application Features

- Modern dark-themed dashboard.
- Home page with project overview.
- Interactive CO concentration prediction.
- Sensor and environmental input fields.
- Tuned Extra Trees model integration.
- Model evaluation metrics.
- Feature information and project methodology.
- Input validation and error handling.

### Run the Application Locally

**1. Clone the repository**

```bash
git clone https://github.com/alishaachhabra/Air-Quality-Analysis-and-Prediction.git
```

**2. Navigate to the project directory**

```bash
cd Air-Quality-Analysis-and-Prediction
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Start the Streamlit application**

```bash
python -m streamlit run app.py
```

**5. Open the application**

Visit the local address displayed in your terminal:

```text
http://localhost:8501
```

Make sure the `air_quality_output` directory contains the trained model and its corresponding feature and metrics files.

## Project Structure

```text
Air_Quality_ML_Project/
│
├── air_quality_output/
│   ├── best_air_quality_model.pkl
│   ├── feature_names.pkl
│   ├── feature_metadata.json
│   ├── model_metrics.json
│   ├── tuned_extra_trees_model.pkl
│   ├── tuned_feature_names.pkl
│   └── tuned_model_metrics.json
│
├── AirQualityUCI.csv
├── AirQualityUCI.xlsx
├── Air_Quality_Analysis_Upgraded.ipynb
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Key Learning Outcomes

- Data cleaning and preprocessing.
- Exploratory data analysis.
- Regression modelling and algorithm comparison.
- Chronological model evaluation.
- Time-series cross-validation.
- Hyperparameter optimisation.
- Model interpretability using SHAP.
- Integrating machine learning models with Streamlit.
- Git and GitHub version control.

## Limitations

- Predictions depend on the quality and distribution of the training data.
- Model estimates should not be interpreted as official air-quality measurements.
- The application estimates CO(GT) from supplied feature values; it is not necessarily a forecast of future air quality.

## Disclaimer

This project is developed for educational and analytical purposes. Its predictions should not replace official air-quality monitoring systems, environmental assessments, or health-related advice.

## Author

**Alisha Chhabra**

B.Tech CSE – Data Science

GitHub: https://github.com/alishaachhabra

---

*Air Quality Analysis and Prediction Using Machine Learning*