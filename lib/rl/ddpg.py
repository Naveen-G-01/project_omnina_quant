# lib/rl/ddpg.py
#
# Deep Deterministic Policy Gradient (DDPG) implementation for HAQ.
# Based on the architecture used for Hardware-Aware Automated Quantization.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from scipy.stats import truncnorm

# ==========================================
# Neural Network Models (Actor & Critic)
# ==========================================

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=300, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = self.fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = self.fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def fanin_init(self, size, fanin=None):
        fanin = fanin or size[0]
        v = 1. / np.sqrt(fanin)
        return torch.Tensor(size).uniform_(-v, v)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out)) # Action bound between 0 and 1
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=300, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc11 = nn.Linear(nb_states, hidden1)
        self.fc12 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc11.weight.data = self.fanin_init(self.fc11.weight.data.size())
        self.fc12.weight.data = self.fanin_init(self.fc12.weight.data.size())
        self.fc2.weight.data = self.fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def fanin_init(self, size, fanin=None):
        fanin = fanin or size[0]
        v = 1. / np.sqrt(fanin)
        return torch.Tensor(size).uniform_(-v, v)

    def forward(self, xs):
        x, a = xs
        out = self.fc11(x) + self.fc12(a)
        out = self.relu(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# ==========================================
# Replay Buffer
# ==========================================

class Memory:
    def __init__(self, limit):
        self.limit = limit
        self.data = deque(maxlen=limit)

    def append(self, state, action, reward, next_state, terminal=False):
        self.data.append((state, action, reward, next_state, terminal))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def __len__(self):
        return len(self.data)

# ==========================================
# DDPG Agent
# ==========================================

class DDPG:
    def __init__(self, nb_states, nb_actions, args):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.args = args
        self.discount = args.discount
        self.tau = args.tau
        self.is_training = True

        # Exploration Noise params
        self.init_delta = args.init_delta
        self.delta_decay = args.delta_decay
        self.delta = self.init_delta

        # Memory
        self.memory = Memory(limit=args.rmsize)

        # Networks
        self.actor = Actor(self.nb_states, self.nb_actions, args.hidden1, args.hidden2, args.init_w)
        self.actor_target = Actor(self.nb_states, self.nb_actions, args.hidden1, args.hidden2, args.init_w)
        self.critic = Critic(self.nb_states, self.nb_actions, args.hidden1, args.hidden2, args.init_w)
        self.critic_target = Critic(self.nb_states, self.nb_actions, args.hidden1, args.hidden2, args.init_w)

        # Copy target weights
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # Cuda configuration
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr_a)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr_c)
        
        # Loss tracking
        self.value_loss = 0.0
        self.policy_loss = 0.0

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def reset(self, observation):
        pass # Optional reset for noise processes if using OU Noise

    def random_action(self):
        action = np.random.uniform(0., 1., self.nb_actions)
        return action

    def select_action(self, s_t, episode):
        state = self.to_tensor(np.array([s_t]))
        action = self.actor(state).squeeze(0).cpu().data.numpy()

        if self.is_training:
            # Truncated normal noise for exploration (as used in AMC/HAQ)
            noise = truncnorm.rvs(-action / self.delta, (1.0 - action) / self.delta, loc=0., scale=self.delta)
            action = action + noise
            action = np.clip(action, 0., 1.)

            # Decay exploration noise
            if episode > self.args.warmup:
                self.delta *= self.delta_decay
                self.delta = max(self.delta, 0.01) # Floor for noise
                
        return action

    def observe(self, reward, s_t, s_t1, a_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, reward, s_t1, done)

    def update_policy(self):
        if len(self.memory) < self.args.bsize:
            return

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample(self.args.bsize)

        state_batch = self.to_tensor(np.array(state_batch))
        action_batch = self.to_tensor(np.array(action_batch))
        reward_batch = self.to_tensor(np.array(reward_batch)).unsqueeze(1)
        next_state_batch = self.to_tensor(np.array(next_state_batch))
        terminal_batch = self.to_tensor(np.array(terminal_batch).astype(np.float32)).unsqueeze(1)

        # ---------------------- Critic Update ----------------------
        with torch.no_grad():
            next_action_batch = self.actor_target(next_state_batch)
            q_next = self.critic_target([next_state_batch, next_action_batch])
            target_q = reward_batch + (1.0 - terminal_batch) * self.discount * q_next

        current_q = self.critic([state_batch, action_batch])
        
        criterion = nn.MSELoss()
        critic_loss = criterion(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---------------------- Actor Update ----------------------
        actor_loss = -self.critic([state_batch, self.actor(state_batch)]).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ---------------------- Target Update ----------------------
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        # Store for logging
        self.value_loss = critic_loss.item()
        self.policy_loss = actor_loss.item()

    def get_value_loss(self):
        return self.value_loss

    def get_policy_loss(self):
        return self.policy_loss

    def get_delta(self):
        return self.delta

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), f'{output_dir}/actor.pth.tar')
        torch.save(self.critic.state_dict(), f'{output_dir}/critic.pth.tar')

    def load_model(self, output_dir):
        if os.path.exists(f'{output_dir}/actor.pth.tar'):
            self.actor.load_state_dict(torch.load(f'{output_dir}/actor.pth.tar'))
        if os.path.exists(f'{output_dir}/critic.pth.tar'):
            self.critic.load_state_dict(torch.load(f'{output_dir}/critic.pth.tar'))

    def to_tensor(self, ndarray):
        tensor = torch.from_numpy(ndarray).float()
        if self.use_cuda:
            tensor = tensor.cuda()
        return tensor
