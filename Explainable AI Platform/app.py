import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained Logistic Regression model and scaler
lr_model = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the original training data (for one-hot encoding and feature consistency)
df = pd.read_csv('credit_risk_dataset.csv')

# Perform one-hot encoding on the categorical variables as it was done during training
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Get the list of columns used during training (for consistent input handling)
trained_columns = df_encoded.drop('loan_status', axis=1).columns

# Streamlit UI
st.title("Credit Risk Prediction")

# Input fields for the user
person_age = st.number_input('Age', min_value=18, max_value=100, value=25)
person_income = st.number_input('Income ($)', min_value=1000, max_value=1000000, value=50000)
person_emp_length = st.number_input('Employment Length (in years)', min_value=0, max_value=50, value=5)

loan_amnt = st.number_input('Loan Amount ($)', min_value=500, max_value=35000, value=10000)
loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, max_value=30.0, value=10.0)
loan_percent_income = st.number_input('Loan as Percent of Income (%)', min_value=0.0, max_value=100.0, value=20.0)

cb_person_cred_hist_length = st.number_input('Credit History Length (in years)', min_value=1, max_value=30, value=5)

# One-hot encoding for categorical variables (same as during training)
person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE'])
loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'BUSINESS'])
loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
cb_person_default_on_file = st.selectbox('Default on File', ['Y', 'N'])

# Convert categorical input to one-hot encoding
categorical_input = {
    'person_home_ownership_RENT': 1 if person_home_ownership == 'RENT' else 0,
    'person_home_ownership_MORTGAGE': 1 if person_home_ownership == 'MORTGAGE' else 0,
    'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
    'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
    'loan_intent_BUSINESS': 1 if loan_intent == 'BUSINESS' else 0,
    'loan_grade_B': 1 if loan_grade == 'B' else 0,
    'loan_grade_C': 1 if loan_grade == 'C' else 0,
    'loan_grade_D': 1 if loan_grade == 'D' else 0,
    'loan_grade_E': 1 if loan_grade == 'E' else 0,
    'loan_grade_F': 1 if loan_grade == 'F' else 0,
    'loan_grade_G': 1 if loan_grade == 'G' else 0,
    'cb_person_default_on_file_Y': 1 if cb_person_default_on_file == 'Y' else 0
}

# Combine numerical inputs and one-hot encoding inputs
input_data = np.array([
    person_age, person_income, person_emp_length,
    loan_amnt, loan_int_rate, loan_percent_income,
    cb_person_cred_hist_length
] + list(categorical_input.values()))

# Ensure the input data matches the training feature set (handle missing columns)
input_data_dict = dict(zip(trained_columns, input_data))
aligned_input_data = np.array([input_data_dict.get(col, 0) for col in trained_columns]).reshape(1, -1)

# Separate numerical features (to scale) and categorical features (do not scale)
numerical_columns = [
    'person_age', 'person_income', 'person_emp_length', 
    'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
    'cb_person_cred_hist_length'
]

# Extract numerical values for scaling
numerical_data = aligned_input_data[0, :len(numerical_columns)]

# Apply scaling only to the numerical features
numerical_data_scaled = scaler.transform(numerical_data.reshape(1, -1))

# Combine the scaled numerical features with the categorical features (not scaled)
final_input_data = np.concatenate([numerical_data_scaled, aligned_input_data[0, len(numerical_columns):].reshape(1, -1)], axis=1)

# Prediction
if st.button('Predict Default'):
    prediction = lr_model.predict(final_input_data)
    if prediction == 1:
        st.write("The person is likely to **default**.")
    else:
        st.write("The person is likely to **not default**.")
