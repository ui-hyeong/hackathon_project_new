from sklearn.metrics import f1_score


def f1score(y, y_hat) :
    y = y.cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    lab_list = []
    out_list = []
    for i in range(len(y_hat)):
      lab = max(y_hat[i])
      if lab == y_hat[i][0]:
        lab = 0
      elif lab == y_hat[i][1]:
        lab = 1
      elif lab == y_hat[i][2]:
        lab = 2
      elif lab == y_hat[i][3]:
        lab = 3
      elif lab == y_hat[i][4]:
        lab = 4
      elif lab == y_hat[i][5]:
        lab = 5
      out_list.append(lab)
    for i in range(len(y)):
      lab_list.append(y[i])
    f1 = f1_score(out_list,lab_list,average='macro')

    return f1