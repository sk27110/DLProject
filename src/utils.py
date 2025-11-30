import torch


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask