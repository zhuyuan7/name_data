import pandas as pd
import numpy  as np
import random

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader

from transformers import  AdamW,get_linear_schedule_with_warmup

from dataset import CustomDataset
from model import Custom_model
from arg import arg_parser 
from sub import *


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


def main(argp):
    dataset_ex = CustomDataset(csv_file=argp.raw_txt)
    print(len(dataset_ex))
    train_size = int(len(dataset_ex)*0.8)
    valid_size = len(dataset_ex) - train_size
    
    
    train_set, valid_set = data.random_split(dataset_ex, [train_size, valid_size])
    print(len(train_set))
    print(len(valid_set))

    train_loader = DataLoader(train_set, batch_size=argp.batch, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=argp.batch, shuffle=True, drop_last=True)

    model = Custom_model()
    model.cuda()

    optimizer = AdamW(model.parameters(),
                    lr = argp.lr, 
                    eps = argp.eps)


    epochs = argp.epochs
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
        

    model.zero_grad()
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_loss = 0

        model.train()
            
        
        for step, batch in enumerate(train_loader):
            
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch


            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_train_loss = total_loss / len(train_loader)            

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
            
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")


        t0 = time.time()
        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        target_label = []
        prediction_label = []

        for batch in valid_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():   
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # calculate accuracy
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

            # target label vs. predicted_label 
            for j in label_ids:
                target_label.append(j)
            
            for i in logits:
                test_label = np.argmax(i)
                prediction_label.append(test_label)

        # save result data
        new_data =pd.DataFrame({'target_label':target_label,"predicted_label":prediction_label})

        # convert label_idx to label_text
        convert_to_text(new_data)

        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0))) 
    print("")
    print("Training complete!")


if __name__ == '__main__':
    argp = arg_parser()
    main(argp)
 

