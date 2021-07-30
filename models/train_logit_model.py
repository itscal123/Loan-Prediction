from sklearn.linear_model import LogisticRegression
import pickle
from get_data import X_train_norm, y_train
import os
    

if not os.path.isfile("logit_l2.pkl"):
    l2 = LogisticRegression(penalty="l2", solver="saga", random_state=42, max_iter=300)
    l2.fit(X_train_norm, y_train)
    with open("logit_l2.pkl", "wb") as f:
        pickle.dump(l2, f)

if not os.path.isfile("logit_l1.pkl"):
    l1 = LogisticRegression(penalty="l1", solver="saga", random_state=42, max_iter=300)
    l1.fit(X_train_norm, y_train)
    with open("logit_l1.pkl", "wb") as f:
        pickle.dump(l1, f)