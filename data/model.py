import torch 
import torch.nn.functional as F

from dataset import CustomDataset
from arg import arg_parser 

from transformers import BertForSequenceClassification

argp = arg_parser()

class Custom_model(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        self.dataset_ex = CustomDataset(csv_file=argp.raw_txt)
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=argp.num_labels)


    def forward(self,input_ids, attention_mask=None, token_type_ids=None,
                labels=None ):
        output = self.model(input_ids,attention_mask=attention_mask,
            token_type_ids=token_type_ids,labels=labels)

        return output

