import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class emoClassifer(torch.nn.Module):
    def __init__(self, bert: BertModel, num_class: int, device: torch.device):
        super().__init__()
        self.bert = bert
        self.H = bert.config.hidden_size
        # 여기서 모델의 신경망을 설계함
        self.W_hy = torch.nn.Linear(self.H, num_class)  # (H, 3)
        self.to(device)

    def forward(self, X: torch.Tensor):
        """
        :param X:
        :return:
        """
        # tokenizer를 통해 출력된 토큰들을 pre-training된 bert모델에 입력하여 모델이 학습한 히든벡터형식으로 출력시킴
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert(input_ids, token_type_ids, attention_mask)[0]

        return H_all

    def predict(self, X):
        H_all = self.forward(X)  # N, L, H
        # 버트는 cls토큰과 unkown토큰을 출력하는데 문장의 맥락을 파악한 cls토큰을 사용하기 위해 아래와 같이 정의함
        H_cls = H_all[:, 0, :]  # 첫번째(cls)만 가져옴 (N,H)
        # N,H  H,3 -> N,3

        y_hat = self.W_hy(H_cls)
        return y_hat  # N,3

    def training_step(self, X, y):
        '''
        :param X:
        :param y:
        :return: loss
        '''
        # 크로스 엔트로피를 사용하여 로스값을 출력함
        # y = torch.LongTensor(y)
        y_pred = self.predict(X)
        y_pred = F.softmax(y_pred, dim=1)
        # loss
        loss = F.cross_entropy(y_pred, y).sum()
        return loss

