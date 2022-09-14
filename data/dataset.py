
import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader,TensorDataset, RandomSampler


from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

class CustomDataset(Dataset): 
  def __init__(self, csv_file):

    self.all_data = pd.read_csv(csv_file)  # "/home/joo/name_data/data/names/all_data.csv"
    self.name = self.all_data ["name"]
    self.label = self.all_data ['label']


    self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    # input_ids = tokenizer(n, padding='max_length')
    # name = all_data["name"]
    # label = all_data['label']

       

  def __len__(self):
    return len(self.name)

  def __getitem__(self, idx): # 토크나이져로 바로 뽑아
   
    n = self.name[idx]
    label = self.label[idx]
    input_ids = self.tokenizer(n, padding='max_length', max_length=20, truncation=True)

    test_inputs = torch.tensor(input_ids['input_ids'])
    test_masks = torch.tensor(input_ids['attention_mask'])
    test_labels = torch.tensor(label)
   


    return test_inputs, test_masks, test_labels

    

if __name__ == "__main__":
    dataset_ex = CustomDataset(csv_file="/home/joo/name_data/data/names/all_data.csv")
    # data_loder = DataLoader(dataset_ex, batch_size=2, shuffle=True)
    # for test_inputs in data_loder:
    #      print(test_inputs)
    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    a,b, c = dataset_ex.__getitem__(0)
  
    print(a)
    print(b)
    print(c)
    # print(data_loder)
    
    

    


