import torch.nn as nn
from transformers import DistilBertForSequenceClassification


class CustomDistilBERTHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.2):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Dropout(p=dropout_prob), nn.ReLU(), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        return self.head(x)
    

class CustomDistilBERTModel(DistilBertForSequenceClassification):
    def __init__(self, distilbert_model, custom_head):
        super().__init__()
        self.distilbert = distilbert_model
        self.custom_head = custom_head

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        custom_head_output = self.custom_head(pooled_output)
        return custom_head_output