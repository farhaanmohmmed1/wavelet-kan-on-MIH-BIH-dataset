import wfdb
import numpy as np
def scale_per_beat(x, eps=1e-8):
    
    x = x.astype(np.float32)
    mn = x.min(axis=1, keepdims=True)   
    mx = x.max(axis=1, keepdims=True)   
    return (x - mn) / (mx - mn + eps)   


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

classes=dict(
    N=['N','L','R','e','j'],
     S= ['A', 'a', 'J', 'S'],
      V= ['V', 'E'],
    F= ['F'],
   
)
reverse_classes = {
    symbol: class_name
    for class_name, symbols in classes.items()
    for symbol in symbols
}
class_to_number = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
}
x_train=[]
y_train=[]

for i in DS1:
    annotation=wfdb.rdann(f'G:/project/dl_sem_pro/Wav-KAN/data/{i}',extension='atr')

    record=wfdb.rdrecord(f'G:/project/dl_sem_pro/Wav-KAN/data/{i}')

    data=record.p_signal[:,0]

    data[~np.isfinite(data)] = np.nan


    mean_value = np.nanmean(data)


    data = np.nan_to_num(data, nan=mean_value)

    anno=annotation.symbol
    sam=annotation.sample

    j=0
    
    while(j<len(sam)):
        if anno[j] in reverse_classes:
            if sam[j]-90>=0 and sam[j]+110<=len(data):
                x_train.append(np.array(data[sam[j]-90:sam[j]+110]))
                y_train.append(class_to_number[reverse_classes[anno[j]]])
        j+=1
x_train=np.array(x_train)
y_train=np.array(y_train)

x_test=[]
y_test=[]

for i in DS2:
    annotation=wfdb.rdann(f'G:/project/dl_sem_pro/Wav-KAN/data/{i}',extension='atr')

    record=wfdb.rdrecord(f'G:/project/dl_sem_pro/Wav-KAN/data/{i}')

    data=record.p_signal[:,0]

    data[~np.isfinite(data)] = np.nan


    mean_value = np.nanmean(data)


    data = np.nan_to_num(data, nan=mean_value)

    anno=annotation.symbol
    sam=annotation.sample

    j=0
    
    while(j<len(sam)):
        if anno[j] in reverse_classes:
            if sam[j]-90>=0 and sam[j]+110<=len(data):
                x_test.append(np.array(data[sam[j]-90:sam[j]+110]))
                y_test.append(class_to_number[reverse_classes[anno[j]]])
        j+=1
x_test=np.array(x_test)
y_test=np.array(y_test)

x_train = scale_per_beat(x_train)
x_test  = scale_per_beat(x_test)


print(x_train, x_test.shape)
print(x_train.min(), x_train.max())   
print(x_test.min(), x_test.max())     
print(np.isnan(x_train).any(), np.isnan(x_test).any())  