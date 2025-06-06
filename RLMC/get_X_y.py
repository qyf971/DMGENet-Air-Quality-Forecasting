import shutil
import os
import numpy as np

model_groups = {
    'proposed': ['model_D', 'model_N', 'model_C', 'model_S'],
}


predict_lens = [1, 2, 3, 4, 5, 6]
targets = ['PM25']

for group_name, models in model_groups.items():
    for predict_len in predict_lens:
        for target in targets:
            dst_folder = f'./data/Run_{run}/{group_name}/{predict_len}/{target}'
            os.makedirs(dst_folder, exist_ok=True)

            base_dir = f'./results_Beijing_12/{predict_len}/model_D'
            val_X = os.path.join(base_dir, 'val_X.npy')
            val_y = os.path.join(base_dir, 'val_y.npy')
            val_y_inverse = os.path.join(base_dir, 'val_y_inverse.npy')
            test_X = os.path.join(base_dir, 'test_X.npy')
            test_y = os.path.join(base_dir, 'test_y.npy')
            test_y_inverse = os.path.join(base_dir, 'test_y_inverse.npy')

            if os.path.exists(val_X): shutil.copy(val_X, dst_folder)
            if os.path.exists(val_y): shutil.copy(val_y, dst_folder)
            if os.path.exists(val_y_inverse): shutil.copy(val_y_inverse, dst_folder)
            if os.path.exists(test_X): shutil.copy(test_X, dst_folder)
            if os.path.exists(test_y): shutil.copy(test_y, dst_folder)
            if os.path.exists(test_y_inverse): shutil.copy(test_y_inverse, dst_folder)

            for i in ['val', 'test']:
                pred_all = []
                pred_inverse_all = []

                for model in models:
                    model_base_dir = f'./results_Beijing_12/{predict_len}/{model}'
                    val_pred = os.path.join(model_base_dir, f'{i}_predictions.npy')
                    val_pred_inverse = os.path.join(model_base_dir, f'{i}_predictions_inverse.npy')

                    if os.path.exists(val_pred) and os.path.exists(val_pred_inverse):
                        pred_all.append(np.load(val_pred))
                        pred_inverse_all.append(np.load(val_pred_inverse))
                    else:
                        print(f"Warning: Missing file for model {model} in {i} predictions")

                if pred_all and pred_inverse_all:
                    pred_all = np.stack(pred_all, axis=1)
                    pred_inverse_all = np.stack(pred_inverse_all, axis=1)
                    np.save(os.path.join(dst_folder, f'{i}_predictions_all.npy'), pred_all)
                    np.save(os.path.join(dst_folder, f'{i}_predictions_inverse_all.npy'), pred_inverse_all)

print('Done!')
