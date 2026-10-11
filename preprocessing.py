import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

def preprocess_record(data, fs):
    b, a = butter(2, 0.5 / (fs / 2), btype='high')
    data = filtfilt(b, a, data)
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230"
]

DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234"
]

classes = dict(
    N=['N', 'L', 'R', 'e', 'j'],
    S=['A', 'a', 'J', 'S'],
    V=['V', 'E'],
    F=['F'],
)
reverse_classes = {
    symbol: class_name
    for class_name, symbols in classes.items()
    for symbol in symbols
}
class_to_number = {"N": 0, "S": 1, "V": 2, "F": 3}

DATA_DIR = '/home/itadmin/Videos/dl/data'


def load_data(records):
    x, y, g = [], [], []          # g = record (patient) id for every beat

    for i in records:
        annotation = wfdb.rdann(f'{DATA_DIR}/{i}', extension='atr')
        record = wfdb.rdrecord(f'{DATA_DIR}/{i}')
        if 'MLII' not in record.sig_name:
            print(f"Skipping {i}: {record.sig_name}")
            continue
        index = record.sig_name.index('MLII')
        data = record.p_signal[:, index]

        data[~np.isfinite(data)] = np.nan
        mean_value = np.nanmean(data)
        data = np.nan_to_num(data, nan=mean_value)
        data = preprocess_record(data, record.fs)

        anno = annotation.symbol
        sam = annotation.sample

        for j in range(len(sam)):
            if anno[j] in reverse_classes:
                if sam[j] - 90 >= 0 and sam[j] + 110 <= len(data):
                    x.append(data[sam[j] - 90:sam[j] + 110])
                    y.append(class_to_number[reverse_classes[anno[j]]])
                    g.append(int(i))

    return np.array(x, dtype=np.float32), np.array(y), np.array(g)


x_train, y_train, g_train = load_data(DS1)
x_test, y_test, g_test = load_data(DS2)

class_names = {0: "N", 1: "S", 2: "V", 3: "F"}

def count_classes(y, title):
    labels, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\n{title} (total = {total})")
    for lab, c in zip(labels, counts):
        print(f"  {class_names[lab]}: {c:6d}  ({100 * c / total:.2f}%)")

def count_per_record(y, g, title):
    print(f"\n{title}: beats per record  [N, S, V, F]")
    for rec in np.unique(g):
        cnt = np.bincount(y[g == rec], minlength=4)
        print(f"  {rec}: {cnt.tolist()}")

count_classes(y_train, "DS1 (train)")
count_classes(y_test, "DS2 (test)")
count_per_record(y_train, g_train, "DS1")

np.savez("dataset.npz",
         x_train=x_train, y_train=y_train, g_train=g_train,
         x_test=x_test, y_test=y_test, g_test=g_test)