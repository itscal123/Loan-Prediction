from sklearn.ensemble import RandomForestClassifier
from get_data import X_train, y_train
import pickle
import os

for n_estimators in (100, 250, 500):
    for criterion in ("gini", "entropy"):
        if not os.path.isfile("random_forest{}_{}.pkl"):
            rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, n_jobs=-1)
            rfc.fit(X_train, y_train)
            with open("random_forest{}_{}.pkl".format(n_estimators, criterion), "wb") as f:
                pickle.dump(rfc, f)    
