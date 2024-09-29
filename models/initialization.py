import torch.nn as nn
import torch.nn.init as init

def initialize_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def initialize_weights_zeros(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.zeros_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def initialize_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)