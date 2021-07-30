import streamlit as st
import logit 
import knn
import random_forest
import naive_bayes
import neural_network
from get_data import variables, df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import text

train_set, test_set = train_test_split(df, test_size=0.1)
y_train, y_test = train_set["MIS_Status"], test_set["MIS_Status"]
X_train, X_test = train_set.loc[:, df.columns != "MIS_Status"], test_set.loc[:, df.columns != "MIS_Status"]
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
pca = PCA(n_components=0.95)
X_test_reduced = pca.fit_transform(X_test)


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