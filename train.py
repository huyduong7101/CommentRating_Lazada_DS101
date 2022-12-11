import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, glob, sys, shutil
import argparse
from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers

from src import CommentDataset, Logger
from src import Trainer, BertModel

def create_cfg():
    parser = argparse.ArgumentParser(description="")

    # path & log
    parser.add_argument("--log_folder", type=str, default="./checkpoints")
    parser.add_argument("--version_name", type=str, default="baseline")
    parser.add_argument("--save_weight_frequency", type=int, default=1)
    parser.add_argument("--print_frequency", type=int, default=-1)

    # model
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base')

    # data
    parser.add_argument('--stopword_path', type=str, default='./data/stopwords.txt')
    parser.add_argument('--data_path', type=str, default="./data/data.csv")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--fold", type=int, default=0)

    # optimizer
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    # others
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    cfg = parser.parse_args()
    return cfg

def main(cfg):
    # load params
    cfg.log_path = os.path.join(cfg.log_folder, cfg.version_name)
    os.makedirs(cfg.log_path, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_name)

    # build dataset & dataloader
    data_df = pd.read_csv(cfg.data_path)
    train_df = data_df[data_df.fold != cfg.fold]
    valid_df = data_df[data_df.fold == cfg.fold]
    
    train_dataset = CommentDataset(cfg, train_df, tokenizer)
    valid_dataset = CommentDataset(cfg, valid_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                num_workers=cfg.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                num_workers=cfg.num_workers, pin_memory=True)

    # build model
    model = BertModel(cfg)
    model.to(cfg.device)
    
    # build optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.num_epochs)

    # build logger
    logger = None
    if cfg.print_frequency == -1:
        cfg.print_frequency = max(1,len(train_dataset) // 10)
        
    print("=="*20)
    print(f"Number of datapoints | train: {len(train_dataset)} | valid: {len(valid_dataset)}")
    print("Save checkpoint to ", cfg.log_path)
    print(f"Model: {cfg.model_name} | Version: {cfg.version_name}")
    print(f"Save checkpoint to {cfg.log_path}")
    print(f"Config: {cfg.__dict__}")
    print("=="*20)

    # build trainer
    trainer = Trainer(cfg=cfg, logger=logger, loss=loss, optimizer=optimizer, lr_scheduler=lr_scheduler)
    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    cfg = create_cfg()
    main(cfg)
