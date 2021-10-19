import torch
from tqdm.autonotebook import tqdm

from hackathon_project.f1_score import f1score
from torch.nn import functional as F


def train_test(EPOCHS=None, model=None, train_dataloader=None, test_dataloader=None, optimizer=None, log_interval=None,
          max_grad_norm=None, scheduler=None):
    for e_idx, epoch in enumerate(range(EPOCHS)):
        losses = list()
        test_f1 = 0.0
        train_f1 = 0.0

        model.train() # 이거선언해야 모델의 신경망이 수정됨
        for b_idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad() # zero_grad를 선언함으로써 역전파를 통한 기울기를 초기화 시킴
            X, y = batch
            loss = model.training_step(X, y) # 손실함수를 통해 loss값을 구함

            loss.backward()  # 역전파를 통해 기울기를 구함
            # 기울기 소실과 폭주 문제를 해결하기 위해 기울기의 절대값이 특정값을 넘어가게 될 경우 기울기의 절대값을 줄여줌
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()  # gradient step
            scheduler.step()  # resetting the gradients.

            # ========== 여기부터 학습 정확도 측정 ===================
            y_hat = model.predict(X)
            y_hat = F.softmax(y_hat, dim=1)
            _, max_indices = torch.max(y_hat, 1)

            train_f1 = f1score(y, max_indices)
            losses.append(loss.item())
            avg_loss = (sum(losses) / len(losses))
            if (b_idx+1) % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e_idx+1, b_idx + 1, avg_loss, train_f1/(b_idx+1)))
        print("epoch {} avg_loss {} train acc {} ".format(e_idx+1, avg_loss, train_f1))

        model.eval() # 이거 선언해야 신경망이 안바뀜
        for b_idx, batch in enumerate(tqdm(test_dataloader)):
            X, y = batch
            y_hat = model.predict(X)
            y_hat = F.softmax(y_hat, dim=1)
            _, max_indices = torch.max(y_hat, 1)
            test_f1 += f1score(y, max_indices)  # accuracy
        print(f"epoch {e_idx+1} test acc {test_f1/(b_idx+1)}" )

    torch.save(model, 'model.pth')