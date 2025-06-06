import os
import numpy as np
import pandas as pd


def calculate_smape(y_true, y_pred):
    epsilon = 1e-8
    smape = 100 * np.mean(
        np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon),
        axis=(1, 2)
    )
    return smape


def calculate_mape(y_true, y_pred):
    epsilon = 1e-8
    mape = 100 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)), axis=(1, 2))
    return mape


def calculate_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=(1, 2))
    return mae


def compute_error_metrics(true_file_path, pred_file_path, metric='smape'):
    y_true = np.load(true_file_path)
    y_pred = np.load(pred_file_path)

    if metric == 'smape':
        error_values = calculate_smape(y_true, y_pred)
    elif metric == 'mape':
        error_values = calculate_mape(y_true, y_pred)
    elif metric == 'mae':
        error_values = calculate_mae(y_true, y_pred)
    else:
        raise ValueError("Unsupported metric. Choose from 'smape', 'mape', 'mae'.")

    return error_values


def shift_error_values(error_values):
    shifted_error_values = np.roll(error_values, shift=1)
    shifted_error_values[0] = error_values[-1]
    return shifted_error_values


models = {
    'proposed': ['model_D', 'model_N', 'model_C', 'model_S'],
}

predict_lens = [1, 2, 3, 4, 5, 6]
targets = ['PM25']
errors = ['mae', 'mape', 'smape']

for group_name, model_list in models.items():
    for predict_len in predict_lens:
        for target in targets:
            for error in errors:
                val_dfs = []
                test_dfs = []
                any_file_found = False

                for model in model_list:
                    in_dir = f'./results_Beijing_12/{predict_len}/{model}'

                    try:
                        val_y_file = os.path.join(in_dir, 'val_y_inverse.npy')
                        val_pred_file = os.path.join(in_dir, 'val_predictions_inverse.npy')
                        test_y_file = os.path.join(in_dir, 'test_y_inverse.npy')
                        test_pred_file = os.path.join(in_dir, 'test_predictions_inverse.npy')

                        if not (os.path.exists(val_y_file) and os.path.exists(val_pred_file)):
                            print(f"Missing files for validation in {in_dir}")
                            continue
                        if not (os.path.exists(test_y_file) and os.path.exists(test_pred_file)):
                            print(f"Missing files for testing in {in_dir}")
                            continue

                        val_error_values = compute_error_metrics(val_y_file, val_pred_file, metric=error)
                        test_error_values = compute_error_metrics(test_y_file, test_pred_file, metric=error)

                        val_history_errors = shift_error_values(val_error_values)
                        test_history_errors = shift_error_values(test_error_values)

                        val_df = pd.DataFrame(val_history_errors, columns=[model])
                        test_df = pd.DataFrame(test_history_errors, columns=[model])

                        val_dfs.append(val_df)
                        test_dfs.append(test_df)
                        any_file_found = True

                    except FileNotFoundError as e:
                        print(f"File not found for model {model} in {in_dir}: {e}")

                if not any_file_found:
                    print(f"No valid files found for group {group_name}, predict_len {predict_len}, target {target}, error {error}")
                    continue

                if val_dfs:
                    combined_val_df = pd.concat(val_dfs, axis=1)
                if test_dfs:
                    combined_test_df = pd.concat(test_dfs, axis=1)

                out_dir = f'data/Run_{run}/{group_name}/{predict_len}/{target}'
                os.makedirs(out_dir, exist_ok=True)


                if val_dfs:
                    combined_val_df.to_csv(os.path.join(out_dir, f'combined_val_{error}_history_errors.csv'), index=False)
                if test_dfs:
                    combined_test_df.to_csv(os.path.join(out_dir, f'combined_test_{error}_history_errors.csv'), index=False)

print('Done!')
