from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pickle
from get_data import X_train, y_train
import os

"""
for n_neighbors in (2, 5, 10):
    for weight in ("uniform", "distance"):
        for p in (1, 2):
            if not os.path.isfile("knn{}_{}_{}.pkl".format(n_neighbors, weight, p)):
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight, p=p, n_jobs=-1)
                knn.fit(X_train, y_train)  
                with open("knn{}_{}_{}.pkl".format(n_neighbors, weight, p), "wb") as f:
                    pickle.dump(knn, f)
"""

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
for n_neighbors in (2, 5, 10):
    for weight in ("uniform", "distance"):
        for p in (1, 2):
            if not os.path.isfile("knn_pca{}_{}_{}.pkl".format(n_neighbors, weight, p)):
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight, p=p, n_jobs=-1)
                knn.fit(X_reduced, y_train)  
                with open("knn_pca{}_{}_{}.pkl".format(n_neighbors, weight, p), "wb") as f:
                    pickle.dump(knn, f)