import os
import numpy as np
import pandas as pd
import torch
import time

from DDPG_RLMC import RLMC_env, DDPG, ReplayBuffer
from utils.metrics import metric
from utils.tools import adjust_learning_rate_RLMC, EarlyStopping


class Exp:
    def __init__(self, state_dim, action_dim, gamma, lr_actor, lr_critic, tau, hidden_dim, episodes, max_steps, batch_size, replay_buffer_size,
                 val_X, val_history_errors, val_y, val_predictions_all,
                 test_X, test_history_errors, test_y, test_predictions_all, test_y_inverse, test_predictions_inverse_all,
                 results_folder):
        self.episodes = episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.action_dim = action_dim
        # dataset
        self.val_X = val_X
        self.val_history_errors = val_history_errors
        self.val_y = val_y
        self.val_predictions_all = val_predictions_all
        self.test_X = test_X
        self.test_history_errors = test_history_errors
        self.test_y = test_y
        self.test_predictions_all = test_predictions_all
        
        self.test_y_inverse = test_y_inverse
        self.test_predictions_inverse_all = test_predictions_inverse_all
        # env and agent
        self.env = RLMC_env(val_X, val_history_errors, val_y, val_predictions_all, action_dim)
        self.agent = DDPG(state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, tau)
        self.buffer = ReplayBuffer(replay_buffer_size)
    
        self.results_folder = results_folder

    def train(self):
        val_mse_list = []
        test_mse_list = []
        episode_time = []
        
        early_stopping = EarlyStopping(patience=5, verbose=True, path=os.path.join(self.results_folder, 'best_model.pth'))
        for episode in range(self.episodes):
            start_time = time.time()
            observation, error = self.env.reset()
            episode_reward = 0
            for step in range(self.max_steps):
                action = self.agent.act(observation, error)
                next_observation, next_error, reward, done, _ = self.env.step(action)
                self.buffer.add(observation, error, action, reward, next_observation, next_error, done)
                if len(self.buffer) > batch_size:
                    self.agent.update(self.buffer.sample(self.batch_size))
                observation, error = next_observation, next_error
                episode_reward += reward
                if done:
                    break
            val_mse, test_mse = self.vali()
            val_mse_list.append(val_mse)
            test_mse_list.append(test_mse)
            end_time = time.time()
            episode_time.append(end_time - start_time)
            print("Episode: {:<5}Cost time: {:<10.2f}Reward: {:<15.7f}Val MSE: {:<15.6f}Test MSE: {:<15.7f}".format(episode + 1, end_time - start_time, episode_reward, val_mse, test_mse))
            
            adjust_learning_rate_RLMC(self.agent.actor_optimizer, episode + 1, self.agent.lr_actor)
            adjust_learning_rate_RLMC(self.agent.critic_optimizer, episode + 1, self.agent.lr_critic)
            
            early_stopping(test_mse, self.agent.actor)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        mse_df = pd.DataFrame({
            'val_mse': val_mse_list,
            'test_mse': test_mse_list,
            'episode_time': episode_time
        })
        mse_df.to_csv(os.path.join(self.results_folder, 'loss.csv'), index=False)
        
        best_model_path = os.path.join(self.results_folder, 'best_model.pth')
        self.agent.actor.load_state_dict(torch.load(best_model_path))

    def evaluate(self, X, history_errors, y, predictions_all):
        
        assert len(X) == len(history_errors), "X and history_errors lengths do not match"
        assert len(X) == len(predictions_all), "X and predictions_all lengths do not match"
        assert len(X) == len(y), "X and y lengths do not match"

        pred = []
        true = []
        for i in range(len(X)):
            data = X[i]
            error = history_errors[i]
            action = self.agent.act(data, error)
            bm_pred = predictions_all[i]
            final_pred = np.sum(action.reshape(self.action_dim, 1, 1) * bm_pred, axis=0)
            target = y[i]
            pred.append(final_pred)
            true.append(target)
        pred = np.array(pred)
        true = np.array(true)
        mse = np.mean((pred - true) ** 2)
        return mse


    def vali(self):
        val_mse = self.evaluate(self.val_X, self.val_history_errors, self.val_y, self.val_predictions_all)
        test_mse = self.evaluate(self.test_X, self.test_history_errors, self.test_y, self.test_predictions_all)
        return val_mse, test_mse



    def test(self):
        pred = []
        true = []
        weights = []
        with torch.no_grad():
            for i in range(len(self.test_X)):
                data = self.test_X[i]
                error = self.test_history_errors[i]
                action = self.agent.act(data, error)
                bm_pred = self.test_predictions_inverse_all[i]
                final_pred = np.sum(action.reshape(self.action_dim, 1, 1) * bm_pred, axis=0)
                target = self.test_y_inverse[i]
                pred.append(final_pred)
                true.append(target)
                weights.append(action)
        pred = np.array(pred)
        true = np.array(true)
        weights = np.array(weights).squeeze(-1)
        weights = pd.DataFrame(weights)

        # saving results
        np.save(os.path.join(self.results_folder, './final_pred.npy'), pred)
        # saving weights
        weights.to_csv(os.path.join(self.results_folder, 'weights.csv'), index=False)
        # computing metrics
        metrics = metric(pred, true)
        # printing metrics
        print(f'test_MAE:{metrics[0]:.4f}, test_RMSE:{metrics[1]:.4f}, test_IA:{metrics[2]:.4f}')
        # saving metrics
        metrics_df = pd.DataFrame([metrics], columns=[f'test_MAE', f'test_RMSE', f'test_IA'])
        metrics_df.to_csv(os.path.join(self.results_folder, 'test_metrics.csv'), index=False)


state_dim = 10
action_dim = 4
hidden_dim = 64
batch_size = 64
lr_actor = 1e-4
lr_critic = 1e-3
tau = 0.001
gamma = 0.9
replay_buffer_size = 10000
episodes = 20
max_steps = 100


for run in range(3, 4):
    for predict_len in [1, 2, 3, 4, 5, 6]:
        for target in ['PM25']:
            for part in ['proposed']:

                base_dir = f'./data/{part}/{predict_len}/'
                results_folder = f'./results/{part}/{predict_len}/'
                os.makedirs(results_folder, exist_ok=True)

                # val_dataset
                val_X = np.load(os.path.join(base_dir, 'val_X.npy'))
                val_history_errors = pd.read_csv(os.path.join(base_dir, 'combined_val_mae_history_errors.csv')).values.astype('float32')
                val_y = np.load(os.path.join(base_dir, 'val_y.npy'))
                val_predictions_all = np.load(os.path.join(base_dir, 'val_predictions_all.npy'))

                # test_dataset
                test_X = np.load(os.path.join(base_dir, 'test_X.npy'))
                test_history_errors = pd.read_csv(os.path.join(base_dir, 'combined_test_mae_history_errors.csv')).values.astype('float32')
                test_y = np.load(os.path.join(base_dir, 'test_y.npy'))
                test_predictions_all = np.load(os.path.join(base_dir, 'test_predictions_all.npy'))
                
                test_y_inverse = np.load(os.path.join(base_dir, 'test_y_inverse.npy'))
                test_predictions_inverse_all = np.load(os.path.join(base_dir, 'test_predictions_inverse_all.npy'))

                exp = Exp(state_dim, action_dim, gamma, lr_actor, lr_critic, tau, hidden_dim, episodes, max_steps, batch_size, replay_buffer_size,
                          val_X, val_history_errors, val_y, val_predictions_all,
                          test_X, test_history_errors, test_y, test_predictions_all,
                          test_y_inverse, test_predictions_inverse_all,
                          results_folder)
                print('training start')
                exp.train()
                print('training end')
                print('testing start') 
                exp.test()
                print('testing end')
