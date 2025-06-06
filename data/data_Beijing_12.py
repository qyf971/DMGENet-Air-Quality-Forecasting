import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def data_standardization():
    data_list = []
    columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'DEWP', 'wd', 'WSPM']
    data_dir = '../dataset/Beijing_12/cleaned_data'
    for filename in os.listdir(data_dir):
        data = pd.read_csv(os.path.join(data_dir, filename), usecols=columns).to_numpy()
        data_list.append(data)
    data_all = np.concatenate(data_list, axis=0)
    scaler = StandardScaler()
    scaler.fit(data_all)

    PM25_mean = scaler.mean_[0]
    PM25_std = scaler.scale_[0]
    PM10_mean = scaler.mean_[1]
    PM10_std = scaler.scale_[1]

    data_norm_list = []
    for data in data_list:
        data_norm = scaler.transform(data)
        data_norm_list.append(data_norm)

    data_norm = np.array(data_norm_list)

    return data_norm, PM25_mean, PM25_std, PM10_mean, PM10_std

def generate_input_target(data, seq_len, predict_len, target):
    num_nodes, T, F = data.shape
    num_samples = T - seq_len - predict_len + 1

    X = np.zeros((num_samples, num_nodes, seq_len, F))
    # y = np.zeros((num_samples, num_nodes, predict_len, 1))
    y = np.zeros((num_samples, num_nodes, 1, 1))

    target_index = 0  # 默认为 PM25 的索引
    if target == 'PM10':
        target_index = 1  # 假设 PM10 是第二个特征

    for i in range(num_samples):
        X[i] = data[:, i:i + seq_len]
        # y[i] = data[:, i + seq_len:i + seq_len + predict_len, target_index:target_index + 1]
        y[i] = data[:, i + seq_len + predict_len - 1:i + seq_len + predict_len, target_index:target_index + 1]

    return X, y



def split_data(X, y, train_ratio, val_ratio, test_ratio, random_state=None):
    num_samples = X.shape[0]

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.0")

    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    if random_state is not None:
        np.random.seed(random_state)
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    predict_len_list = [1, 2, 3, 4, 5, 6]
    seq_len = 24
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    for predict_len in predict_len_list:
        Output_dir = Path('./dataset/Beijing_12/one_step/train_val_test_data/622') / f'{predict_len}'
        Output_dir.mkdir(parents=True, exist_ok=True)

        data_norm, PM25_mean, PM25_std, PM10_mean, PM10_std = data_standardization()

        scaler_PM25 = np.array([PM25_mean, PM25_std])
        scaler_PM10 = np.array([PM10_mean, PM10_std])
        np.save(Output_dir / 'scaler_PM25.npy', scaler_PM25)
        np.save(Output_dir / 'scaler_PM10.npy', scaler_PM10)

        print(f"input_len:{seq_len} predict_len:{predict_len} 数据处理开始")
        for target in ['PM25', 'PM10']:
            X, y = generate_input_target(data_norm, seq_len, predict_len, target)
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, train_ratio, val_ratio, test_ratio)
            np.savez_compressed(Output_dir / f'train_{target}.npz', X=X_train, y=y_train)
            np.savez_compressed(Output_dir / f'val_{target}.npz', X=X_val, y=y_val)
            np.savez_compressed(Output_dir / f'test_{target}.npz', X=X_test, y=y_test)
            print(f"{target}_train_X: {X_train.shape}  {target}_train_y: {y_train.shape}")
            print(f"{target}_val_X:   {X_val.shape}    {target}_val_y:   {y_val.shape}")
            print(f"{target}_test_X:  {X_test.shape}   {target}_test_y:  {y_test.shape}")
        print(f"input_len:{seq_len} predict_len:{predict_len} 数据处理完毕")