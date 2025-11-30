import torch


def generate_square_subsequent_mask(sz):
    """
    Создаёт маску для Transformer decoder,
    чтобы предотвратить "заглядывание" на будущие токены
    """
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask