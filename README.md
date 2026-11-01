\# Air Quality Analysis and Prediction Using Machine Learning



\## Overview



This project analyzes air quality data and predicts carbon monoxide (CO) concentration using machine learning. It includes a Streamlit web application that allows users to enter environmental measurements and obtain a model-based prediction.



\## Objectives



\* Analyze air quality and environmental measurements.

\* Train and evaluate machine learning regression models.

\* Select a model using chronological evaluation.

\* Provide an interactive interface for making predictions.


## Project Structure

```text
Air_Quality_ML_Project/
├── air_quality_output/
│   ├── best_air_quality_model.pkl
│   ├── feature_names.pkl
│   ├── feature_metadata.json
│   └── model_metrics.json
├── AirQualityUCI.csv
├── AirQualityUCI.xlsx
├── app.py
├── requirements.txt
└── README.md
```

* `app.py`: Streamlit application for air quality prediction.
* `AirQualityUCI.csv` and `AirQualityUCI.xlsx`: Dataset files.
* `air_quality_output/`: Saved model, feature information, and evaluation metrics.
* `requirements.txt`: Python dependencies.
* `README.md`: Project documentation.




\## Technology Stack



\* Python

\* Pandas and NumPy

\* Scikit-learn

\* Streamlit

\* Joblib

\* Matplotlib



\## Machine Learning



The project evaluates multiple regression algorithms and selects the best-performing model based on chronological holdout results.



The currently selected model is Extra Trees.



\* R²: 0.835

\* MAE: 0.3667

\* RMSE: 0.5461



These metrics represent the project's chronological evaluation results and are not a guarantee of real-world predictive accuracy.



\## Application Features



\* Interactive environmental input fields.

\* Machine learning-based CO prediction.

\* Prediction results displayed through a dashboard.

\* Model evaluation information.

\* A note explaining that predictions are estimates, not official air quality measurements.



\## Run Locally



1\. Clone or download this repository.



2\. Install the required packages:



&#x20;  `pip install -r requirements.txt`



3\. Start the application:



&#x20;  `python -m streamlit run app.py`



4\. Open the local URL displayed in the terminal.



\## Disclaimer



This application is intended for educational and analytical purposes. Its predictions should not replace official air quality monitoring measurements.



\## Author



Alisha Chhabra



