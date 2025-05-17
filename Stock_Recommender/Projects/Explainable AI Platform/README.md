# **Credit Risk Prediction App**

This project is a **Credit Risk Prediction** model built using **Logistic Regression**, with a **Streamlit** frontend to allow users to input their data and receive predictions on whether a person is likely to default on a loan. 

The app uses machine learning and data preprocessing techniques to analyze a person's financial profile and predict the likelihood of a loan default based on various features such as income, loan amount, employment length, and credit history.

---

## **Features**:

- **User Input**: The app takes input for a variety of features such as age, income, loan amount, and more.
- **Preprocessing**: The app automatically processes categorical variables using **one-hot encoding** and scales numerical features using **StandardScaler**.
- **Prediction**: After the input is processed, the app uses a pre-trained **Logistic Regression model** to predict whether the user is likely to default on a loan.
- **Interactive Frontend**: Built using **Streamlit**, the app provides an interactive web interface where users can easily input data and view results.

---

## **Technologies Used**:
- **Streamlit**: For building the frontend.
- **scikit-learn**: For training the Logistic Regression model and data preprocessing.
- **pandas**: For handling and processing the data.
- **numpy**: For numerical operations.
- **joblib**: For saving and loading the trained model and scaler.

---

## **Getting Started**:

### **Prerequisites**:

Before you start, ensure you have the following installed:

- Python 3.6 or higher
- **pip** (for installing Python packages)

### **Installing Dependencies**:

Clone the repository and install the necessary packages:

```bash
git clone <your-repo-url>
cd <your-project-directory>
pip install -r requirements.txt
```

Where `requirements.txt` includes:

```
streamlit
pandas
numpy
scikit-learn
joblib
```

### **Running the App**:

1. **Run the following command** in your terminal:

   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`) to start using the app.

---

## **Model Overview**:

### **Training the Model**:

- The model was trained using a **Logistic Regression** algorithm on a dataset containing features such as:
  - **person_age**: Age of the person
  - **person_income**: Income of the person
  - **loan_amnt**: Loan amount
  - **loan_int_rate**: Interest rate on the loan
  - **loan_percent_income**: Loan amount as a percentage of the person's income
  - **cb_person_cred_hist_length**: Length of the person's credit history
  - **person_home_ownership**: Type of home ownership (Rent, Own, Mortgage)
  - **loan_intent**: Purpose of the loan (Personal, Education, Medical, Business)
  - **loan_grade**: Credit grade of the loan
  - **cb_person_default_on_file**: Whether the person has defaulted on a loan before
  
### **Preprocessing**:
- **One-hot encoding** was applied to the categorical features.
- **Feature scaling** was applied to numerical features using **StandardScaler**.

### **Model Evaluation**:
The model's performance was evaluated using:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report** (precision, recall, and F1-score)

---

## **Project Structure**:

```
credit-risk-prediction/
│
├── app.py              # Streamlit frontend app
├── credit_risk_dataset.csv  # The dataset used for training the model
├── lr_model.pkl        # Trained Logistic Regression model
├── scaler.pkl          # StandardScaler used for feature scaling
├── requirements.txt    # List of Python dependencies
├── README.md           # Project documentation (this file)
```

---

## **Contributing**:

Feel free to fork this repository, create a branch, and submit a pull request if you'd like to contribute to the project.
