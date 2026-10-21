import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

keras.utils.set_random_seed(42)

PROTOCOL = "paper"    # "paper": balance ALL data, then random 80/20 split (likely what the paper did, optimistic)
                      # "clean": balance train only, test on the official mitbih_test.csv (no leakage)
BALANCE = True        # oversample/undersample classes to N_PER_CLASS
N_PER_CLASS = 20000
VAR_SORT = True       # reorder the 187 time points by variance (fit on train only)


@keras.utils.register_keras_serializable()
class KANLayer(keras.layers.Layer):
    """phi(x) = w_b * silu(x) + w_s * spline(x)  (B-spline, as in the original KAN)."""

    def __init__(self, out_features, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0), **kw):
        super().__init__(**kw)
        self.out_features, self.grid_size = out_features, grid_size
        self.spline_order, self.grid_range = spline_order, tuple(grid_range)

    def build(self, input_shape):
        in_f = int(input_shape[-1])
        k, g = self.spline_order, self.grid_size
        h = (self.grid_range[1] - self.grid_range[0]) / g
        grid = (np.arange(-k, g + k + 1) * h + self.grid_range[0]).astype("float32")
        self.grid = tf.constant(np.tile(grid, (in_f, 1)))                  # (in, g+2k+1)
        self.n_basis = g + k
        self.base_w = self.add_weight(name="base_w", shape=(in_f, self.out_features),
                                      initializer="he_uniform")
        self.spline_w = self.add_weight(name="spline_w", shape=(in_f, self.n_basis, self.out_features),
                                        initializer=keras.initializers.RandomNormal(stddev=0.1 / g))
        self.spline_scaler = self.add_weight(name="spline_scaler", shape=(in_f, self.out_features),
                                             initializer="he_uniform")
        self.in_f = in_f

    def b_splines(self, x):
        x = tf.expand_dims(x, -1)                                          # (B, in, 1)
        grid = self.grid
        bases = tf.cast((x >= grid[:, :-1]) & (x < grid[:, 1:]), x.dtype)  # (B, in, g+2k)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
            bases = left + right
        return bases                                                       # (B, in, g+k)

    def call(self, x):
        base = tf.matmul(tf.nn.silu(x), self.base_w)
        w = self.spline_w * tf.expand_dims(self.spline_scaler, 1)          # (in, nb, out)
        spline = tf.matmul(tf.reshape(self.b_splines(x), (-1, self.in_f * self.n_basis)),
                           tf.reshape(w, (self.in_f * self.n_basis, self.out_features)))
        return base + spline

    def get_config(self):
        cfg = super().get_config()
        cfg.update(out_features=self.out_features, grid_size=self.grid_size,
                   spline_order=self.spline_order, grid_range=self.grid_range)
        return cfg


train_df = pd.read_csv("G:/project/dl_sem_pro/Wav-KAN/data/mitbih_train.csv", header=None)
test_df = pd.read_csv("G:/project/dl_sem_pro/Wav-KAN/data/mitbih_test.csv", header=None)

x_train, y_train = train_df.iloc[:, :-1].values.astype("float32"), train_df.iloc[:, -1].values.astype(int)
x_test, y_test = test_df.iloc[:, :-1].values.astype("float32"), test_df.iloc[:, -1].values.astype(int)

def balance(x, y):
    rng = np.random.RandomState(42)
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0], N_PER_CLASS, replace=(y == c).sum() < N_PER_CLASS)
        for c in range(5)
    ])
    idx = rng.permutation(idx)
    return x[idx], y[idx]


if PROTOCOL == "paper":
    x_all, y_all = np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])
    if BALANCE:
        x_all, y_all = balance(x_all, y_all)
    x_tr, x_test, y_tr, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)
else:
    tr, va = train_test_split(np.arange(len(x_train)), test_size=0.2, random_state=42, stratify=y_train)
    x_tr, y_tr = x_train[tr], y_train[tr]
    x_va, y_va = x_train[va], y_train[va]
    if BALANCE:
        x_tr, y_tr = balance(x_tr, y_tr)

print("train counts:", np.bincount(y_tr, minlength=5).tolist())
print("val counts  :", np.bincount(y_va, minlength=5).tolist())
print("test counts :", np.bincount(y_test, minlength=5).tolist())

if VAR_SORT:
    order = np.argsort(x_tr.var(axis=0))
    x_tr, x_va, x_test = x_tr[:, order], x_va[:, order], x_test[:, order]

model = keras.Sequential([
    keras.layers.Input(shape=(187,)),
    KANLayer(128),
    KANLayer(5),
    keras.layers.Softmax(),
])

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
model.fit(
    x_tr, y_tr,
    validation_data=(x_va, y_va),
    epochs=150,
    batch_size=64,
)

pred = model.predict(x_test, batch_size=1024).argmax(1)
print(classification_report(y_test, pred, labels=[0, 1, 2, 3, 4], target_names=["N", "S", "V", "F", "Q"], digits=4, zero_division=0))
print(confusion_matrix(y_test, pred, labels=[0, 1, 2, 3, 4]))
model.save("KAN_mitbih.keras")