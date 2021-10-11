import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# from emoclassfer_2.EMOClassifer import emoClassifer


# bertmodel = BertModel.from_pretrained("beomi/kcbert-large")
# tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-large")
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.__version__)


class emoClassifer(torch.nn.Module):
    def __init__(self, bert: BertModel, num_class: int, device: torch.device):
        super().__init__()
        self.bert = bert
        self.H = bert.config.hidden_size
        self.W_hy = torch.nn.Linear(self.H, num_class)  # (H, 3)
        self.to(device)

    def forward(self, X: torch.Tensor):
        """
        :param X:
        :return:
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert(input_ids, token_type_ids, attention_mask)[0]

        return H_all

    def predict(self, X):
        H_all = self.forward(X)  # N, L, H
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
        # y = torch.LongTensor(y)
        y_pred = self.predict(X)
        y_pred = F.softmax(y_pred, dim=1)
        # loss
        loss = F.cross_entropy(y_pred, y).sum()
        return loss

def Build_X (sents, tokenizer, device):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1).to(device)

def predict(DATA):
    bertmodel = BertModel.from_pretrained("beomi/kcbert-base")
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")

    device = torch.device('cuda:0')
    model = torch.load(r'C:\Users\jeonguihyeong\PycharmProjects\hackathon_project\emoclassfer_2\epoch20.pth', map_location=device)
    # model.eval()
    X = Build_X(DATA, tokenizer, device)
    print(X)
    y_hat = model.predict(X)
    y_hat = F.softmax(y_hat, dim=1)

    return y_hat
print(predict('선생님이 나를 혼냈어'))

#
# 기쁨 0
#
# 슬픔 1
#
# 불안 2
#
# 당황 3
#
# 분노 4
#
# 상처 5