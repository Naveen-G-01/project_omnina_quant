# lib/env/quantize_env.py
#
# Hardware-Aware Automated Quantization (HAQ) Environment
# This environment interfaces with the DDPG agent to evaluate candidate
# bit-width strategies via K-Means weight quantization.

import os
import time
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from lib.utils.utils import AverageMeter, accuracy
from lib.utils.data_utils import get_split_train_dataset
from lib.utils.quantize_utils import quantize_model, kmeans_update_model


class QuantizeEnv:
    def __init__(self, model, pretrained_model, dataset, dataset_root,
                 compress_ratio, n_data_worker, batch_size, args, float_bit=32,
                 is_model_pruned=False):
        self.model = model
        self.pretrained_model = pretrained_model
        self.dataset = dataset
        self.dataset_root = dataset_root
        self.compress_ratio = compress_ratio
        self.float_bit = float_bit
        self.args = args
        self.is_model_pruned = is_model_pruned

        # Dataloaders for fine-tuning and validation during the RL search
        self.train_loader, self.val_loader, self.n_class = get_split_train_dataset(
            dataset_name=self.dataset, batch_size=batch_size, n_worker=n_data_worker,
            val_size=args.val_size, train_size=args.train_size, data_root=dataset_root)

        self.criterion = nn.CrossEntropyLoss().cuda()

        # Architecture extraction
        self.quantizable_idx = []
        self.bound_list = []
        self._build_index()

        self.n_macs = []
        self.n_params = []
        self._get_model_info()

        self.layer_embedding = self._build_state_embedding()

        self.reset()

    def _build_index(self):
        # Identify layers eligible for quantization
        for i, m in enumerate(self.model.modules()):
            if type(m) in [nn.Conv2d, nn.Linear]:
                self.quantizable_idx.append(i)
                self.bound_list.append((self.args.min_bit, self.args.max_bit))

    def _get_model_info(self):
        # Gather parameter counts and approximate MACs for the state space
        for m in self.model.modules():
            if type(m) in [nn.Conv2d, nn.Linear]:
                self.n_params.append(m.weight.numel())
                if isinstance(m, nn.Conv2d):
                    # Rough MAC approximation based on spatial kernel expansion
                    self.n_macs.append(m.weight.numel() * 10) 
                else:
                    self.n_macs.append(m.weight.numel())

    def _build_state_embedding(self):
        # Constructs a continuous state feature vector per layer for the DDPG agent
        layer_embedding = []
        max_params = max(self.n_params) if self.n_params else 1
        max_macs = max(self.n_macs) if self.n_macs else 1

        for i in range(len(self.quantizable_idx)):
            state = [
                i / len(self.quantizable_idx),
                self.n_params[i] / max_params,
                self.n_macs[i] / max_macs,
                self.bound_list[i][0] / self.float_bit,
                self.bound_list[i][1] / self.float_bit
            ]
            layer_embedding.append(np.array(state))
        return np.array(layer_embedding)

    def reset(self):
        self.cur_ind = 0
        self.strategy = []
        self.model.load_state_dict(deepcopy(self.pretrained_model))
        return self.layer_embedding[self.cur_ind]

    def step(self, action):
        # Map continuous action [0, 1] to discrete bit widths
        action = action[0]
        min_bit, max_bit = self.bound_list[self.cur_ind]
        action = min_bit + action * (max_bit - min_bit)
        action = max(min_bit, min(max_bit, action))
        
        # Round to nearest bit-width mapping for K-Means
        action = int(np.round(action))
        self.strategy.append(action)

        done = False
        info = {}
        reward = 0

        self.cur_ind += 1
        if self.cur_ind == len(self.quantizable_idx):
            done = True
            info = self._evaluate()
            reward = self._compute_reward(info)
            obs = self.layer_embedding[0] # Loop back to start for interface consistency
        else:
            obs = self.layer_embedding[self.cur_ind]

        return obs, reward, done, info

    def _evaluate(self):
        # Apply the proposed quantization strategy
        centroids_dict = quantize_model(self.model, self.quantizable_idx, self.strategy, 
                                        mode='cpu', quantize_bias=False, is_pruned=self.is_model_pruned)
        kmeans_update_model(self.model, self.quantizable_idx, centroids_dict)

        if self.args.finetune_flag:
            self._finetune()

        # Evaluate accuracy
        acc = self._validate()

        # Calculate size/compression ratio
        total_params = sum(self.n_params)
        quantized_size = sum([self.n_params[i] * self.strategy[i] for i in range(len(self.n_params))])
        w_ratio = quantized_size / (total_params * self.float_bit)

        info = {
            'accuracy': acc,
            'w_ratio': w_ratio,
        }
        return info

    def _compute_reward(self, info):
        # HAQ Reward Function: accuracy minus penalty for exceeding target compression ratio
        acc = info['accuracy']
        w_ratio = info['w_ratio']
        
        reward = acc
        if w_ratio > self.compress_ratio:
            # Heavy penalty if the policy exceeds the target model footprint
            reward -= (w_ratio - self.compress_ratio) * 100

        return reward

    def _finetune(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.finetune_lr, momentum=0.9, weight_decay=1e-4)
        for epoch in range(self.args.finetune_epoch):
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _validate(self):
        self.model.eval()
        top1 = AverageMeter()
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs)
                prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
        return top1.avg
