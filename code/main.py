import torch
import torch.nn.functional as F
import pandas as pd
import numpy  as np
import random
import datetime
import time

from torch.utils.data import DataLoader
import torch.utils.data as data

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

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(),
                    lr = argp.lr, # 학습률
                    eps = argp.eps # 0으로 나누는 것을 방지하기 위한 epsilon 값
                    )

    # 에폭수
    epochs = argp.epochs

    # 총 훈련 스텝 : 배치반복 횟수 * 에폭
    total_steps = len(train_loader) * epochs

    # 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
        

    # 그래디언트 초기화
    model.zero_grad()

    # 에폭만큼 반복
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # 시작 시간 설정
        t0 = time.time()

        # 로스 초기화
        total_loss = 0

        # 훈련모드로 변경
        model.train()
            
        # 데이터로더에서 배치만큼 반복하여 가져옴
        for step, batch in enumerate(train_loader):
            # 경과 정보 표시
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)
            
            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            # Forward 수행                
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            
            # 로스 구함
            loss = outputs[0]

            # 총 로스 계산
            total_loss += loss.item()

            # Backward 수행으로 그래디언트 계산
            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 그래디언트를 통해 가중치 파라미터 업데이트
            optimizer.step()

            # 스케줄러로 학습률 감소
            scheduler.step()

            # 그래디언트 초기화
            model.zero_grad()

        # 평균 로스 계산
        avg_train_loss = total_loss / len(train_loader)            

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
            
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        #시작 시간 설정
        t0 = time.time()

        # 평가모드로 변경
        model.eval()

        # 변수 초기화
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        target_label = []
        prediction_label = []
        # 데이터로더에서 배치만큼 반복하여 가져옴
        for batch in valid_loader:
            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)
            
            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch
            
            # 그래디언트 계산 안함
            with torch.no_grad():     
                # Forward 수행
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            
            # 로스 구함
            logits = outputs[0]
            
            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            
            # 출력 로짓과 라벨을 비교하여 정확도 계산
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
        new_data =pd.DataFrame({'target_label':target_label,"predict_label":prediction_label})

        # convert label_idx to label_text
        convert_to_text(new_data)

        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    

    print("")
    print("Training complete!")



if __name__ == '__main__':
    argp = arg_parser()
    main(argp)
 

