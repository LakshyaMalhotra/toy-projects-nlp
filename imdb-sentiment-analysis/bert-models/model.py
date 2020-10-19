# model.py
import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        # fetch the model from the BERT_PATH defined in config.py
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)

        # add a dropout for regularization
        self.bert_drop = nn.Dropout(0.3)

        # linear layer for the output
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # BERT in its default settings returns two outputs:
        # last hidden state and output of the BERT pooler layer
        # output of the pooler layer is of the size: (batch_size, hidden_size)
        # hidden size is 768 in our case
        _, out_pooler = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )

        # pass through the dropout layer
        bert_out = self.bert_drop(out_pooler)

        # pass through the fc layer
        output = self.fc(bert_out)

        # return output
        return output
