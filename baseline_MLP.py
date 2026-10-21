import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

keras.utils.set_random_seed(42)

train_df = pd.read_csv("G:/project/dl_sem_pro/Wav-KAN/data/mitbih_train.csv", header=None)
test_df = pd.read_csv("G:/project/dl_sem_pro/Wav-KAN/data/mitbih_test.csv", header=None)

x_train, y_train = train_df.iloc[:, :-1].values.astype("float32"), train_df.iloc[:, -1].values.astype(int)
x_test, y_test = test_df.iloc[:, :-1].values.astype("float32"), test_df.iloc[:, -1].values.astype(int)

tr, va = train_test_split(np.arange(len(x_train)), test_size=0.2, random_state=42, stratify=y_train)
print("val counts [N,S,V,F,Q]:", np.bincount(y_train[va], minlength=5).tolist())

model = keras.Sequential([
    keras.layers.Input(shape=(187,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(5, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
model.fit(
    x_train[tr], y_train[tr],
    validation_data=(x_train[va], y_train[va]),
    epochs=150,
    batch_size=64,
)

pred = model.predict(x_test, batch_size=1024).argmax(1)
print(classification_report(y_test, pred, labels=[0, 1, 2, 3, 4], target_names=["N", "S", "V", "F", "Q"], digits=4, zero_division=0))
print(confusion_matrix(y_test, pred, labels=[0, 1, 2, 3, 4]))
model.save("MLP_baseline.keras")