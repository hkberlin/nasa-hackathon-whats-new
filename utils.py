import torch

def R2_score(outputs, y):
    y_bar = torch.sum(torch.mean(y))
    SS_tot = torch.sum((y - y_bar)**2)
    SS_res = torch.sum((y - outputs)**2)
    R2 = 1 - SS_res / SS_tot
    return R2