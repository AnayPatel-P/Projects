# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv('credit_risk_dataset.csv')

# Step 2.1: Handle missing values
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

# Step 2.2: Handle outliers
# Remove unrealistic ages
df = df[df['person_age'] <= 100]

# Cap income at 99th percentile
income_cap = df['person_income'].quantile(0.99)
df['person_income'] = np.where(df['person_income'] > income_cap, income_cap, df['person_income'])

# Step 2.3: One-hot encode categorical variables
categorical_cols = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 2.4: Standardize numeric features
num_cols = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length'
]

scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# Step 2.5: Train-test split
X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Step 3.1: Model evaluation
y_pred = lr_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import joblib

# Assuming you already have lr_model and scaler from previous steps
# Save the model and scaler for later use in the Streamlit app
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
