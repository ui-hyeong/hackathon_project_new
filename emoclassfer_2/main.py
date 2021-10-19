from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from emoclassfer_2.EMOClassifer import emoClassifer
from hackathon_project.Builder import Build_X, Build_y
from hackathon_project.SimpleDataset import SimpleDataset
from hackathon_project.train_test import train_test
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

# device = torch.device("cuda:0")

DATA = pd.read_csv('test.csv')

DATA = DATA.values.tolist()




def main():
    # 설정값들 수정하기 위해 위에 몰아서 썻음
    test_size = 0.3
    random_state = 13
    batch_size = 64
    EPOCHS = 10
    learning_rate = 5e-5
    num_class = 6
    max_grad_norm = 1
    warmup_ratio = 0.1
    log_interval = 200

    # 쿠다 사용 가능 여부 확인
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)

    # gpu 사용 코드
    # device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    device = torch.device('cpu')
    print('학습을 진행하는 기기:', device)

    #pre-training 모델 불러오기
    bertmodel = AutoModel.from_pretrained("monologg/kobert")
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    sents = [sent for sent, _ in DATA]
    labels = [label for _, label in DATA]

    # stratify를 label로 설정하면 균일하게 섞어서 검증시 좀더 정확하게 할 수 있음
    x_train, x_test, y_train, y_test = train_test_split(sents, labels,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        stratify=labels,
                                                        random_state=random_state)

    # 데이터 전처리 및 토크나이저 적용
    x_train = Build_X(x_train, tokenizer, device)
    y_train = Build_y(y_train, device)
    x_test = Build_X(x_test, tokenizer, device)
    y_test = Build_y(y_test, device)



    # 파이토치의 dataloader를 사용하기 위한 자연어 강사의 코드.
    # 파이토치에 dataset이라는 기능이 있는데 공부하기 싫어서 이거 씀
    train_dataset = SimpleDataset(x_train, y_train)
    test_dataset = SimpleDataset(x_test, y_test)
    # dataloader를 사용하면 알아서 배치사이즈에 맞게 데이터를 출력해서 순차적으로 학습시킬 수 있음
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    # 학습하기 앞서 모델 정의
    classfer = emoClassifer(bertmodel, num_class=num_class, device=device)
    # 이건 뭔지 기억이 안나는데 아마 bertAdam을 사용 하기 위한 옵티마이저 설정이었음
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classfer.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in classfer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 옵티마이저 설정
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # optimizer = torch.optim.Adam(classfer.parameters(), lr=learning_rate)

    t_total = len(train_dataloader) * EPOCHS
    warmup_step = int(t_total * warmup_ratio)

    # 자연어 강사 슬랙에 올라왔던 스케쥴러. 학습률을 변경시켜 좀더 효율적으로 학습시킴.
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    #------------------------------ train & test ---------------------------------
    print('학습시작')
    train_test(train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               model=classfer,
               EPOCHS=EPOCHS,
               optimizer=optimizer,
               max_grad_norm=max_grad_norm,
               scheduler=scheduler,
               log_interval=log_interval)


if __name__ == '__main__':
    main()
