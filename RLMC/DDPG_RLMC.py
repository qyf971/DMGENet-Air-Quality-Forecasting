import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from _Support.TemporalConvNet import TemporalConvNet


class RLMC_env:
    def __init__(self, data_x, data_error, data_y, bm_pred, action_dim):
        assert len(data_x) == len(data_error) == len(data_y) == len(bm_pred), "All input data must have the same length"
        self.data_x = data_x
        self.data_error = data_error
        self.data_y = data_y
        self.bm_pred = bm_pred
        self.action_dim = action_dim
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        # 随机选择一个起始索引
        # self.current_step = random.randint(0, len(self.data_x) - 1)
        obs, error = self._get_state()
        return obs, error

    def step(self, action):
        reward = self._calculate_reward(action)
        done = self.current_step >= len(self.data_x) - 1
        self.current_step += 1
        next_obs, next_error = self._get_state()
        info = {'error': next_error}
        return next_obs, next_error, reward, done, info

    def _get_state(self):
        obs = self.data_x[self.current_step]
        error = self.data_error[self.current_step]
        return obs, error

    def _calculate_reward(self, action):
        target = self.data_y[self.current_step]
        final_pred = np.sum(action.reshape(self.action_dim, 1, 1) * self.bm_pred[self.current_step], axis=0)
        # return -np.mean((target - final_pred) ** 2)
        return -np.mean(np.abs(target - final_pred))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, observation, error, action, reward, next_observation, next_error, done):
        self.buffer.append((observation, error, action, reward, next_observation, next_error, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        observations, errors, actions, rewards, next_observations, next_errors, dones = zip(*batch)
        return np.array(observations), np.array(errors), np.array(actions), np.array(rewards), np.array(next_observations), np.array(next_errors), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.action_dim = action_dim
        self.causal_cnn = TemporalConvNet(state_dim, [hidden_dim] * 4)
        self.loss_embedding = nn.Embedding(action_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, model_loss):
        batch_size, num_nodes, seq_len, features = obs.size()
        obs = obs.contiguous().view(batch_size, num_nodes*seq_len, features)
        time_series_embedding = self.causal_cnn(obs.transpose(1, 2)).transpose(1, 2)
        model_rank = torch.argsort(model_loss, dim=1)
        loss_embedding = self.loss_embedding(model_rank)
        input_embedding = nn.ReLU()(torch.cat([time_series_embedding, loss_embedding], 1))
        x = self.net(input_embedding)
        return F.softmax(x[:, -self.action_dim:, :], dim=1)   # [B, action_dim, 1]


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.causal_cnn = TemporalConvNet(state_dim, [hidden_dim] * 4)
        self.action_embedding = nn.Linear(1, hidden_dim)
        self.loss_embedding = nn.Embedding(action_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, model_loss, act):
        batch_size, num_nodes, seq_len, features = obs.size()
        obs = obs.contiguous().view(batch_size, num_nodes * seq_len, features)
        time_series_embedding = self.causal_cnn(obs.transpose(1, 2)).transpose(1, 2)
        action_embedding = self.action_embedding(act)
        model_rank = torch.argsort(model_loss, dim=1)
        loss_embedding = self.loss_embedding(model_rank)
        input_embedding = nn.ReLU()(torch.cat([time_series_embedding, loss_embedding, action_embedding], 1))

        x = self.net(input_embedding)
        out = x[:, -1, :]  # 取序列最后一个时间步的输出
        return out


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, tau):
        # 网络
        self.actor = Actor(state_dim, action_dim, hidden_dim).cuda()
        self.critic = Critic(state_dim, action_dim, hidden_dim).cuda()
        self.target_actor = Actor(state_dim, action_dim, hidden_dim).cuda()
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).cuda()
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        # 软更新目标网络参数
        self.soft_update(tau=1)

    def act(self, observation, model_loss):
        observation = torch.FloatTensor(observation).unsqueeze(0).cuda()
        model_loss = torch.FloatTensor(model_loss).unsqueeze(0).cuda()
        action = self.actor(observation, model_loss).detach().cpu().numpy()[0]
        return action  # [action_dim, 1]

    def update(self, batch):
        observations, errors, actions, rewards, next_observations, next_errors, dones = batch
        observations = torch.FloatTensor(observations).cuda()
        errors = torch.LongTensor(errors).cuda()
        actions = torch.FloatTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).cuda().unsqueeze(1)
        next_observations = torch.FloatTensor(next_observations).cuda()
        next_errors = torch.FloatTensor(next_errors).cuda()
        dones = torch.FloatTensor(dones).cuda().unsqueeze(1)

        # Compute target Q values
        next_actions = self.target_actor(next_observations, next_errors)
        target_q_values = self.target_critic(next_observations, next_errors, next_actions)
        target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
        # Compute the current Q values
        current_q_values = self.critic(observations, errors, actions)
        # Compute the loss and update the critic
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor
        actor_loss = -self.critic(observations, errors, self.actor(observations, errors)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络参数
        self.soft_update()

    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
