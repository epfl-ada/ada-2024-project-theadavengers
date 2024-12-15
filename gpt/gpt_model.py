from transformers import GPT2Tokenizer, GPT2Model
import torch
import torch.nn as nn
from datasets import Dataset


class GPTClassifier(nn.Module):
    def __init__(self, gpt_model, num_classes):
        super(GPTClassifier, self).__init__()
        self.gpt = gpt_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.gpt.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
