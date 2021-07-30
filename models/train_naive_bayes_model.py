from sklearn.naive_bayes import GaussianNB
from get_data import X_train_norm, y_train
import pickle
import os

if not os.path.isfile("naive_bayes.pkl"):
    gnb = GaussianNB()
    gnb.fit(X_train_norm, y_train)   
    with open("naive_bayes.pkl", "wb") as f:
        pickle.dump(gnb, f)  