from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os


def train_model(df):
    train_set, test_set = train_test_split(df, test_size=0.1)
    y_train, y_test = train_set["MIS_Status"], test_set["MIS_Status"]
    X_train, X_test = train_set.loc[:, df.columns != "MIS_Status"], test_set.loc[:, df.columns != "MIS_Status"]

    norm = MinMaxScaler().fit(X_train)
    X_train_norm = norm.transform(X_train)
    X_test_norm = norm.transform(X_test)

    if not os.path.isfile("my_keras_model"):
        model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=X_test_norm.shape[1]),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
        ])

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)

        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        history = model.fit(X_train, y_train, epochs=50, callbacks=[early_stopping_cb])
        model.save("my_keras_model")

    return (X_test_norm, y_test)

    def deploy_model(st, X_test, y_test):
        st.header("Neural Network")
        st.write("tba")