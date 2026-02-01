import torch
import torch.nn as nn

def get_loss_function():
    # L1 Loss is preferred for SR as it handles edges better than MSE
    return nn.L1Loss()