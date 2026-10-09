import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import torch


text = "Replace me by any text you'd like."


class bert_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = RobertaTokenizer.from_pretrained('/data3/hly/DAVLT/pretrained_networks/roberta-base')
        self.model = RobertaModel.from_pretrained('/data3/hly/DAVLT/pretrained_networks/roberta-base')
        self.model.to(self.device)


    def forward(self,  text): # one more token
        with torch.no_grad():

            encoded_input = self.tokenizer(text,padding=True,truncation=True,return_tensors='pt').to(self.device)

            output = self.model(**encoded_input)

        return output['last_hidden_state']