import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, sys, shutil
import json
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from utils import set_seed

class Trainer():
    def __init__(self, cfg, logger, loss, optimizer, lr_scheduler):
        self.cfg = cfg
        self.logger = logger
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = self.cfg.device
        
        set_seed(self.cfg.seed)
        
    def fit(self, model: nn.Module, 
            train_loader: torch.utils.data.DataLoader, 
            valid_loader:torch.utils.data.DataLoader):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.save_opt()

        print('Start training ...')
        for self.epoch in range(self.cfg.num_epochs):
            print(f'===== Epoch {self.epoch} ======')
            self.train_one_epoch()
            self.val_one_epoch()

            if self.epoch % self.cfg.save_weight_frequency:
                self.save_weights()
            
    def train_one_epoch(self):
        self.model.train()
        all_preds = []
        all_labels = []

        for i, batch in tqdm(enumerate(self.train_loader)):
            inputs, labels = batch
            for k,v in inputs.items():
                inputs[k] = v.to(self.cfg.device)
            labels = labels.to(self.cfg.device)

            self.optimizer.zero_grad()

            preds = self.model(inputs)
            all_preds.append(preds.argmax(dim=1))
            all_labels.append(labels)

            loss = self.loss(preds, labels)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if i % self.cfg.print_frequency:
                info = 'Training | epoch: {}, step: {}, loss: {}'.format(self.epoch, i, loss)
                print(info)

        metrics = self.compute_metrics(all_labels, all_preds)
        info = 'Training | epoch: {}, metrics: {}'.format(self.epoch, metrics)
        print(info)
        # self.logger.log(info)

    def val_one_epoch(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader)):
                inputs, labels = batch
                for k,v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)

                preds = self.model(inputs)
                all_preds.append(preds.argmax(dim=1))
                all_labels.append(labels)

        metrics = self.compute_metrics(all_labels, all_preds)
        info = 'Validation | epoch: {}, metrics: {}'.format(self.epoch, metrics)
        print(info)
        #self.logger.log(info)

    def compute_metrics(self, labels, preds):
        all_preds = np.concatenate([x.detach().cpu().numpy() for x in preds])
        all_labels = np.concatenate([x.detach().cpu().numpy() for x in labels])
        all_probs = (all_preds > 0.5).astype(int)
        acc = float(accuracy_score(y_true=all_labels, y_pred=all_probs))
        return {"accuracy": acc}

    def predict(self, model: nn.Module, test_loader: torch.utils.data.DataLoader):
        pass

    def save_opt(self):
        path = os.path.join(self.cfg.log_path, 'opt.json')
        with open(path, 'w') as f:
            json.dump(self.cfg.__dict__.copy(), f, indent=2)

    def load_weights(self):
        state_dict = torch.load(self.cfg.load_weights_path, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print(
                    '{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self):
        path = os.path.join(self.cfg.log_path, 'weights_' + str(self.epoch) + '.pth')
        torch.save(self.model.state_dict(), path)