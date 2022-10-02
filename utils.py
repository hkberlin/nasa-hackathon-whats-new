import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def R2_score(outputs, y):
    y_bar = torch.sum(torch.mean(y))
    SS_tot = torch.sum((y - y_bar)**2)
    SS_res = torch.sum((y - outputs)**2)
    R2 = 1 - SS_res / SS_tot
    return R2

def visualize_seq2seq(x, x_prime, prediction, save_path, index):
    original_R2 = R2_score(x_prime, x)
    trained_R2 = R2_score(prediction, x)
    
    print("visualizing:", index)
    # to numpy
    x = x.cpu().numpy()
    x_prime = x_prime.cpu().numpy()
    prediction = prediction.cpu().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    # Bx
    axs[0].plot(x_prime[:, 0], linewidth=1, color='grey', alpha=0.7, label='DSCOVR-Mag')
    axs[0].plot(x[:, 0], linewidth=1, color='black', alpha=1.0, label='Wind-MFI')
    axs[0].plot(prediction[:, 0], linewidth=1, color='red', alpha=1.0, label='Prediction')
    axs[0].grid()
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("Bx")

    # By
    axs[1].plot(x_prime[:, 1], linewidth=1, color='grey', alpha=0.7, label='DSCOVR-Mag')
    axs[1].plot(x[:, 1], linewidth=1, color='black', alpha=1.0, label='Wind-MFI')
    axs[1].plot(prediction[:, 1], linewidth=1, color='red', alpha=1.0, label='Prediction')
    axs[1].grid()
    axs[1].legend(loc="upper right")
    axs[1].set_ylabel("By")

    # Bz
    axs[2].plot(x_prime[:, 2], linewidth=1, color='grey', alpha=0.7, label='DSCOVR-Mag')
    axs[2].plot(x[:, 2], linewidth=1, color='black', alpha=1.0, label='Wind-MFI')
    axs[2].plot(prediction[:, 2], linewidth=1, color='red', alpha=1.0, label='Prediction')
    axs[2].grid()
    axs[2].legend(loc="upper right")
    axs[2].set_ylabel("Bz")

    plt.setp(axs, xlim=[-30, 850])

    fig.suptitle(f'original R2: {original_R2:.4f}, trained R2: {trained_R2:.4f}')
    fig.supxlabel('Time Period (hr)')
    plt.savefig(os.path.join(save_path, f"test_{index}.png"))
    plt.close()