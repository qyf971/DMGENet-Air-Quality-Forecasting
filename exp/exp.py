import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.dataloader_Beijing_12 import Dataloader_Beijing_12

from utils.metrics import metric
from utils.tools import adjust_learning_rate, EarlyStopping, plot_loss


import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


class Exp_model:
    def __init__(self, model_name, model, epoch, learning_rate, target, batch_size, num_workers, dataset, predict_len, patience):
        self.model_name = model_name
        self.model = model.cuda()
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.target = target
        self.patience = patience

        print(f'chosen dataset:{dataset} model_name:{self.model_name} forecasting target:{self.target} predict_len:{predict_len}')

        # results folder
        self.results_folder = f'./results_{dataset}/{predict_len}/{self.model_name}'
        os.makedirs(self.results_folder, exist_ok=True)

        # dataloader
        self.train_dataloader = Dataloader_Beijing_12(f'./dataset/train_val_test_data/{predict_len}/train_{self.target}.npz', 'train', batch_size, num_workers, target)
        self.val_dataloader = Dataloader_Beijing_12(f'./dataset/train_val_test_data/{predict_len}/val_{self.target}.npz', 'val', batch_size, num_workers, target)
        self.test_dataloader = Dataloader_Beijing_12(f'./dataset/train_val_test_data/{predict_len}/test_{self.target}.npz', 'test', batch_size, num_workers, target)
        self.train_loader = self.train_dataloader.get_dataloader()
        self.val_loader = self.val_dataloader.get_dataloader()
        self.test_loader = self.test_dataloader.get_dataloader()

    def val(self, criterion):
        val_loss = []
        test_loss = []

        self.model.eval()
        for i, (features, target) in enumerate(self.val_loader):
            features = features.cuda()
            target = target.cuda()
            pred, true = self.model(features), target
            loss = criterion(pred, true)
            val_loss.append(loss.item())
        val_loss = np.average(val_loss)

        for i, (features, target) in enumerate(self.test_loader):
            features = features.cuda()
            target = target.cuda()
            pred, true = self.model(features), target
            loss = criterion(pred, true)
            test_loss.append(loss.item())
        test_loss = np.average(test_loss)
        self.model.train()

        return val_loss, test_loss

    def train(self):
        criterion = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_loss = []
        val_loss = []
        test_loss = []

        epoch_time = []

        self.model.train()
        time_start = time.time()

        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=os.path.join(self.results_folder, self.model_name + '.pth'))

        for epoch in range(self.epoch):
            adjust_learning_rate(optim, epoch + 1, self.learning_rate)
            epoch_train_loss = []
            epoch_start_time = time.time()
            for i, (features, target) in enumerate(self.train_loader):
                features = features.cuda()
                target = target.cuda()
                optim.zero_grad()
                pred, true = self.model(features), target
                loss = criterion(pred, true)
                epoch_train_loss.append(loss.item())
                loss.backward()
                optim.step()

            epoch_end_time = time.time()

            epoch_time.append(epoch_end_time - epoch_start_time)

            epoch_train_loss = np.average(epoch_train_loss)
            epoch_val_loss, epoch_test_loss = self.val(criterion)
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)
            test_loss.append(epoch_test_loss)
            print(
                "Epoch [{:<3}/{:<3}] cost time:{:.5f} train_loss:{:.5f} val_loss:{:.5f} test_loss:{:.5f}".format(epoch + 1, 100,
                                                                                                         epoch_end_time - epoch_start_time,
                                                                                                         epoch_train_loss,
                                                                                                         epoch_val_loss,
                                                                                                         epoch_test_loss))
            early_stopping(epoch_val_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        time_end = time.time()
        print("training time is {:.2f}seconds，{:.2f}minutes".format(time_end - time_start, (time_end - time_start) / 60))

        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        test_loss = np.array(test_loss)
        epoch_time = np.array(epoch_time)
        train_loss_df = pd.DataFrame(
            {'epoch_time': epoch_time, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})

        plot_loss(train_loss, val_loss, test_loss)

        # train_loss saving
        train_loss_df.to_csv(os.path.join(self.results_folder, 'loss.csv'), index=True, index_label='epoch')

        best_model_path = os.path.join(self.results_folder, self.model_name + '.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def _evaluate(self, dataloader, loader, flag):
        predictions = []
        trues = []
        features_list = []

        for i, (features, target) in enumerate(loader):
            features = features.cuda()
            features_np = features.cpu().numpy()
            features_list.append(features_np)
            with torch.no_grad():
                pred = self.model(features)
            true = target
            predictions.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        all_features = np.concatenate(features_list, axis=0)
        trues = np.concatenate(trues, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        trues = np.squeeze(trues, axis=-1)
        predictions = np.squeeze(predictions, axis=-1)

        print(
            f'{flag}_all_features shape:{all_features.shape}, {flag}_predictions shape:{predictions.shape}, {flag}_trues shape:{trues.shape}')

        np.save(os.path.join(self.results_folder, f'{flag}_X.npy'), all_features)
        np.save(os.path.join(self.results_folder, f'{flag}_y.npy'), trues)
        np.save(os.path.join(self.results_folder, f'{flag}_predictions.npy'), predictions)

        trues_inverse = dataloader.inverse_transform(trues)
        predictions_inverse = dataloader.inverse_transform(predictions)

        np.save(os.path.join(self.results_folder, f'{flag}_y_inverse.npy'), trues_inverse)
        np.save(os.path.join(self.results_folder, f'{flag}_predictions_inverse.npy'), predictions_inverse)

        metrics = metric(predictions_inverse, trues_inverse)

        # printing metrics
        print(f'{flag}_MAE:{metrics[0]:.3f}, {flag}_RMSE:{metrics[1]:.3f}, {flag}_IA:{metrics[2]:.3f}')

        # saving metrics
        metrics_df = pd.DataFrame([metrics], columns=[f'{flag}_MAE', f'{flag}_RMSE', f'{flag}_IA'])
        metrics_df.to_csv(os.path.join(self.results_folder, f'{flag}_metrics.csv'), index=False)

    def test(self):
        self.model.eval()

        self._evaluate(self.train_dataloader, self.train_loader, 'train')
        self._evaluate(self.val_dataloader, self.val_loader, 'val')
        self._evaluate(self.test_dataloader, self.test_loader, 'test')