from typing import List

import torch


# tokenizer를 통해 텍스트 데이터를 인코딩된 input_ids, 토큰의 타입, attention_masking이 된 데이터로 변환시켜줌
def Build_X (sents, tokenizer, device):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1).to(device)

def Build_y (labels: List[int], device):
    y = torch.LongTensor(labels).to(device)
    return y