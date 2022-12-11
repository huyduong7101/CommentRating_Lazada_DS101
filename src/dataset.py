import torch
from torch.utils.data import Dataset

def prepare_input(cfg, text, tokenizer):
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.max_len,
        padding="max_length",
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class CommentDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = prepare_input(self.cfg, row['text'], self.tokenizer)
        label = row['score'] - 1
        return text, label