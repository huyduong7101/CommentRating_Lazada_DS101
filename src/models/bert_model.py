import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

class BertModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg.model_name
        self.model = transformers.AutoModel.from_pretrained(self.model_name)
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        self.head = nn.Sequential(
            nn.Linear(768, cfg.hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
            nn.Softmax()
        )
        
    def forward(self, input):
        input_ids, attention_mask = input["input_ids"], input["attention_mask"]
        
        output = self.model(input_ids, attention_mask)[1]
        output = self.head(output)
        output = output.argmax(axis=1)
        return output
