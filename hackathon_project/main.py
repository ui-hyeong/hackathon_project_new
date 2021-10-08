from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from hackathon_project.EMOClassifer import EMOClassifer
from hackathon_project.Builder import Build_X, Build_y
from hackathon_project.SimpleDataset import SimpleDataset
from hackathon_project.train_test import train_test
from transformers.optimization import get_cosine_schedule_with_warmup




# device = torch.device("cuda:0")

DATA = pd.read_csv('test.csv')

DATA = DATA.values.tolist()




def main():

    test_size = 0.3
    random_state = 13
    batch_size = 27
    EPOCHS = 1
    log_interval = 200
    learning_rate = 6e-6
    num_class = 6

    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)

    device = torch.device("cpu")
    print('학습을 진행하는 기기:', device)
    bertmodel = AutoModel.from_pretrained("monologg/kobert")
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    sents = [sent for sent, _ in DATA]
    labels = [label for _, label in DATA]
    x_train, x_test, y_train, y_test = train_test_split(sents, labels,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        stratify=labels,
                                                        random_state=random_state)

    x_train = Build_X(x_train, tokenizer, device)
    y_train = Build_y(y_train, device)
    x_test = Build_X(x_test, tokenizer, device)
    y_test = Build_y(y_test, device)



    train_dataset = SimpleDataset(x_train, y_train)
    test_dataset = SimpleDataset(x_test, y_test)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    classfer = EMOClassifer(bertmodel, num_class=num_class, device=device)

    optimizer = torch.optim.Adam(params=classfer.parameters(), lr=learning_rate)


    #------------------------------ train & test ---------------------------------

    train_test(train_dataloader = train_dataloader,
               test_dataloader = test_dataloader,
               model = classfer,
               EPOCHS = EPOCHS,
               optimizer = optimizer,
               log_interval = log_interval)

if __name__ == '__main__':
    main()
