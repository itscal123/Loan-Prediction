import streamlit as st

def title():
    st.title("Loan Prediction Analysis and Comparison Between Models")

def welcome():
    st.header("Welcome to the application!")
    st.write("""
    Hello and welcome to my loan repayment prediction app. Before we get
    started let's go through some preliminary information about what this
    app is and how it was made.

    First, let's go through the purpose of this app in the context of machine learing.
    It is no stretch to say that we are currently in the midst of machine learning
    renaissance which has come about due to the increase in computing power and the resurgence
    of neural networks. Not to mention the vast increase in data has allowed for the training
    of algorithms to be much easier than it has ever been. As a result, this application
    aims to utilize and evaluate different algorithms on a popular dataset: loan repayment.

    Next, we discuss the logisitics of the application. This app was constructed
    primarily with Python! Amazingly, Streamlit provides a great platform for data scientists
    to quickly visualize and implement a front-end solution for displaying Python solutions
    in Python. Consequently, all the models constructed were written in Python. 
    """)    

def data(variables):
    # variables is a dataframe that contains the variables and their descriptions
    st.header("Data Used")
    st.write("""
    So let's talk about the dataset we will be building to train our models. The dataset chosen for this
    application is from the U.S. Small Business Administration (SBA). The dataset can be found and 
    downloaded on Kaggle: https://www.kaggle.com/mirbektoktogaraev/should-this-loan-be-approved-or-denied

    The context provided about this dataset is as follows:
    """)

    st.text("""
    The U.S. SBA was founded in 1953 on the principle of promoting and assisting small enterprises
    in the U.S. credit market (SBA Overview and History, US Small Business Administration (2015)).
    Small businesses have been a primary source of job creation in the United States; therfore, 
    fostering small business formation and growth has social benefits by creating job 
    opportunities and reducing unemployment.

    There have been many success stories of start-ups receiving SBA loan guarantees such as FedEx
    and Apple Computer. However, there have also been stories of small businesses and/or start-ups 
    that have defaulted on their SBA-guaranteed loans.""")

    st.write("The provided dataset has 899164 rows (entries) and 27 columns (variables)")
    st.write(variables)    

def data_process():
    st.header("How the data was processed")
    st.write("""
    First, we begin by removing variables that have little to no statistical significance.
    This includes variables such as Name and Loan Number ID which are generally unique values
    that have no impact on whether a borrower was able to pay back their loan. Additionally, 
    feature scaling should be applied to models that require or perform optimally with 
    feature scaling (e.g., logistic regression and neural networks).
    """)
