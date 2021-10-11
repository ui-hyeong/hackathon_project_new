import torch
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer

from hackathon_project.Builder import Build_X
from hackathon_project.examples.BERTclassifer import BERTClassifier
# from hackathon_project.examples.emoclassifer import BERTClassifer






bertmodel = BertModel.from_pretrained("monologg/kobert")
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

DATA =["응애. 나 애기 코린이. "]

print(DATA)
def predict(DATA):

    device = torch.device('cpu')
    model = torch.load(r'C:\Users\jeonguihyeong\PycharmProjects\hackathon_project\epoch5.pth', map_location=device)
    input_ids, token_type_ids, attention_mask  = Build_X(DATA, tokenizer, device)
    y_hat = model.predict(input_ids, token_type_ids, attention_mask)
    y_hat = F.softmax(y_hat, dim=1)
    return y_hat
print(predict(DATA))
