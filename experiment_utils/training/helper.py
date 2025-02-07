import torch


def get_optimizer(optim_name, params, lr, weight_decay=0.0):
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"{optim_name} is not supported.")
    return optimizer
