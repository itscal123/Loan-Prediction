from get_data import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

train_set, test_set = train_test_split(df, test_size=0.1)
y_train, y_test = train_set["MIS_Status"], test_set["MIS_Status"]
X_train, X_test = train_set.loc[:, df.columns != "MIS_Status"], test_set.loc[:, df.columns != "MIS_Status"]
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
pca = PCA(n_components=0.95)
X_test_reduced = pca.fit_transform(X_test)

np.save("test.npy", X_test)
np.save("normalized_test.npy", X_test_norm)
np.save("reduced_test.npy", X_test_reduced)