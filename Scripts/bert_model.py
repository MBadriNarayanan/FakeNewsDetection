from transformers import BertModel
from torch.nn import Dropout, Linear, Module


class BERTModel(Module):
    def __init__(self, model_name, hidden_units, dropout):

        super(BERTModel, self).__init__()

        self.model_name = model_name
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.classes = 3
        self.bert_model = BertModel.from_pretrained(self.model_name)
        self.dropout_layer = Dropout(self.dropout)
        self.linear_layer = Linear(self.hidden_units, self.classes)

    def forward(self, input_id, mask):

        _, output = self.bert_model(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        output = self.dropout_layer(output)
        output = self.linear_layer(output)

        return output
