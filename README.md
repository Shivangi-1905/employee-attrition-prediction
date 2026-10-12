# Employee Attrition Prediction

A machine learning web application that predicts employee attrition using employee-related information. The project explores HR analytics and demonstrates how machine learning can help organizations identify employees who may be at risk of leaving.

## Project Overview

Employee attrition occurs when employees leave an organization. Understanding the factors associated with attrition can help HR teams make more informed decisions about employee retention.

This project uses historical HR data to train a classification model and provides an interactive interface built with Streamlit for making predictions.

## Features

- Predicts whether an employee may leave the organization.
- Interactive web interface built with Streamlit.
- Uses employee information such as age, monthly income, overtime, job role, years at company, and job satisfaction.
- Displays prediction results and attrition probability.
- Includes exploratory data analysis and model comparison.

## Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Dataset:** IBM HR Analytics Employee Attrition dataset

## Machine Learning Approach

The project uses a classification approach to predict employee attrition.

The workflow includes:

1. Loading and inspecting the HR dataset.
2. Cleaning and preparing the data.
3. Exploring employee attributes and attrition patterns.
4. Training and evaluating classification models.
5. Comparing model performance.
6. Saving the trained model for use in the web application.
7. Integrating the model with Streamlit for interactive predictions.

Logistic Regression and Random Forest were explored during model development. Logistic Regression was selected based on the F1-score comparison performed during training.

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset, which contains employee-related attributes and an attrition label.

- **Records:** 1,470 employees
- **Features:** 34 input attributes and 1 target column
- **Target variable:** `Attrition` (`Yes` or `No`)

The dataset is intended for educational and analytical purposes. Its patterns may not represent other organizations or workforces.

## Project Structure

```text
employee-attrition-prediction/
├── app.py
├── requirements.txt
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── models/
│   └── attrition_model.joblib
├── .gitignore
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shivangi-1905/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2. Create a virtual environment

**Windows:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## Limitations

- Predictions depend on the dataset and model used during training.
- The application is an educational demonstration, not a definitive assessment of an employee's intentions.
- Predictions should not be used as the sole basis for employment decisions.
- Model performance and generalizability depend on data quality, feature selection, and evaluation methodology.

## Future Improvements

- Add more employee attributes to the prediction interface.
- Improve model evaluation and handle class imbalance.
- Add feature importance and explainability visualizations.
- Improve the user interface and accessibility.
- Deploy the application online for public access.

## Disclaimer

This project is developed for learning, demonstration, and HR analytics exploration. It should not be used to make automated employment decisions or to judge individual employees.

## Author

**Shivangi**

- GitHub: [@Shivangi-1905](https://github.com/Shivangi-1905)

If you find this project useful, feel free to explore the repository and share suggestions.
