import streamlit as st
import logit 
import knn
import random_forest
import naive_bayes
import neural_network
import text
import numpy as np
import pandas as pd

# Dataframe of the categories and their descriptions
variables = pd.DataFrame({
    "Variable Name": ["LoanNr_ChkDgt", "Name", "City", "State", "Zip", "Bank", "BankState",
                        "NAICS", "ApprovalDate", "ApprovalFY", "Term", "NoEmp", "NewExist",
                        "CreateJob", "RetainedJob", "FranchiseCode", "UrbanRural", "RevLineCr",
                        "LowDoc", "ChgOffData", "DisbursementDate", "DisbursementGross",
                        "BalanceGross", "MIS_Status", "ChgOffPrinGr", "GrAppv", "SBA_Appv"],
    "Description": ["Identifier Primary key", "Borrower name", "Borrower city", "Borrower state",
                        "Borrower zip code", "Bank name", "Bank state", "North American industry classification system code",
                        "Date SBA commitment issued", "Fiscal year of commitment", "Loan term in months", "Number of business employees",
                        "1 = Existing business, 2 = New business", "Number of jobs created", "Number of jobs retained",
                        "Franchise code, (00000 or 00001) = No franchise", "1 = Urban, 2 = rural, 0 = undefined",
                        "Revolving line of credit: Y = Yes, N = No", "LowDoc Loan Program: Y = Yes, N = No", 
                        "The date when a loan is declared to be in default", "Disbursement data", "Amount disbursed",
                        "Gross amount outstanding", "Loan status charged off = CHGOFF, Paid in full = PIF",
                        "Charged-off amount", "Gross amount approved by bank","SBA's guaranteed amount of approved loan"]
})

with open("test.npy", "wb") as f:
    X_test = np.load(f)

with open("reduced_test.npy", "wb") as f:
    X_test_norm = np.load(f)

with open("reduced_test.npy", "wb") as f:
    X_test_reduced = np.load(f)


text.title()
text.welcome()
text.data(variables)
text.data_process()

model_selection = st.radio(
    "Select model",
    ("Logistic Regression (Logit)", "K-Nearest Neighbors (KNN)", "Naive-Bayes", "Random Forest",
    "Neural Network")
)

if model_selection == "Logistic Regression (Logit)":
    logit.output(X_test_norm, y_test)
elif model_selection == "K-Nearest Neighbors (KNN)":
    knn.output(X_test_norm, X_test_reduced, y_test)
elif model_selection == "Naive-Bayes":
    naive_bayes.output(X_test_norm, y_test)
elif model_selection == "Random Forest":
    random_forest.output(X_test, y_test)
else:
    neural_network.output(X_test_norm, y_test)

st.header("Contact Me")
st.write("""
Please feel free to contact me if you have any questions or have spotted an error:
""")
st.write("Email: calvin.kory@gmail.com")
st.write("GitHub: https://github.com/itscal123")