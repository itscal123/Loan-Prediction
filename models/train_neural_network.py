import tensorflow as tf
from tensorflow import keras
import os
from get_data import X_train_norm, y_train

if not os.path.isfile("my_keras_model.h5"):
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=X_train_norm.shape[1]),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.fit(X_train_norm, y_train, validation_split=0.2, epochs=50, callbacks=[early_stopping_cb])
    model.save("my_keras_model.h5")