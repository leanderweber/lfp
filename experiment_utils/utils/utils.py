import logging
import os
import random

import joblib
import numpy as np
import torch
from lfprop.propagation import propagator_zennit

from ..model import models


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def save_rng_state(savepath, device):
    print("SAVING RNG STATES")
    logging.info("SAVING RNG STATES")
    torch_state = torch.get_rng_state()
    joblib.dump(torch_state, os.path.join(savepath, "torch-state.joblib"))

    torch_cuda_state = torch.cuda.get_rng_state(device)
    joblib.dump(torch_cuda_state, os.path.join(savepath, "torch-cuda-state.joblib"))

    np_state = np.random.get_state()
    joblib.dump(np_state, os.path.join(savepath, "np-state.joblib"))

    random_state = random.getstate()
    joblib.dump(random_state, os.path.join(savepath, "random-state.joblib"))


def load_rng_state(savepath, device):
    print("LOADING RNG STATES")
    logging.info("LOADING RNG STATES")
    torch_state = joblib.load(os.path.join(savepath, "torch-state.joblib"))
    torch.set_rng_state(torch_state)

    torch_cuda_state = joblib.load(os.path.join(savepath, "torch-cuda-state.joblib"))
    torch.cuda.set_rng_state(torch_cuda_state, device)

    np_state = joblib.load(os.path.join(savepath, "np-state.joblib"))
    np.random.set_state(np_state)

    random_state = joblib.load(os.path.join(savepath, "random-state.joblib"))
    random.setstate(random_state)


def save_model(model, optimizer, quantizer_model, savepath, filename):
    """
    Saves model and optimizer to given path
    """
    os.makedirs(savepath, exist_ok=True)

    savedict = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        savedict["optimizer_state_dict"] = optimizer.state_dict()

    if quantizer_model is not None:
        savedict["quantizer_state_dict"] = quantizer_model.state_dict()

    torch.save(savedict, os.path.join(savepath, filename))


def load_model(model, optimizer, quantizer_model, savepath, filename):
    """
    Load the model and optimizer from given checkpoint path
    """
    checkpoint = torch.load(os.path.join(savepath, filename))
    if "model_state_dict" in checkpoint:
        # previous save
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "quantizer_state_dict" in checkpoint and quantizer_model is not None:
            quantizer_model.load_state_dict(checkpoint["optimizer_state_dict"])


def save_forward_hook(module, input, output):
    """ """
    # Save module input
    module.saved_output = output[0].detach().cpu().numpy()
    # module.saved_weight = module.weight.data.detach().cpu().numpy()

    # None as return value
    return None


def norm_bw_hook(module, grad_input, grad_output):
    """ """
    if isinstance(grad_input, tuple):
        retlist = []
        for g in grad_input:
            if g is not None:
                retlist.append(
                    g
                    / torch.where(
                        g.abs().max() > 0, g.abs().max(), torch.ones_like(g.abs().max())
                    )
                )
            else:
                retlist.append(None)
        ret = tuple(retlist)
    else:
        if grad_input is not None:
            ret = grad_input / torch.where(
                grad_input.abs().max() > 0,
                grad_input.abs().max(),
                torch.ones_like(grad_input.abs().max()),
            )
        else:
            ret = None
    return ret


def print_bw_hook(module, grad_input, grad_output):
    """ """
    if isinstance(grad_input, tuple):
        for g in grad_input:
            if grad_input[0] is not None:
                print("--------------------------------")
                print(module)
                print(grad_input[0].amin(), grad_input[0].mean(), grad_input[0].amax())
    else:
        if grad_input is not None:

            print("--------------------------------")
            print(module)
            print(grad_input[0].amin(), grad_input[0].mean(), grad_input[0].amax())


def register_backward_normhooks(model):
    """
    Registers backward hooks to nor
    :return:
    """
    # Get layers of model
    layers = models.list_layers(model)

    backward_handles = []
    for layer in layers:
        backward_handles.append(layer.register_backward_hook(norm_bw_hook))

    return backward_handles


def register_backward_printhooks(model):
    """
    Registers backward hooks to nor
    :return:
    """
    # Get layers of model
    layers = models.list_layers(model)

    backward_handles = []
    for layer in layers:
        backward_handles.append(layer.register_backward_hook(print_bw_hook))

    return backward_handles


def register_forward_hooks(model):
    """
    Registers forward hooks to save necessary stuff
    :return:
    """
    # Get layers of model
    layers = models.list_layers(model)

    forward_handles = []
    for layer in layers:
        forward_handles.append(
            layer.register_forward_pre_hook(propagator_zennit.save_input_hook)
        )

    return forward_handles


def remove_forward_hooks(model, forward_handles):
    """
    Removes forward hooks
    :return:
    """
    # Remove forward hooks
    for handle in forward_handles:
        handle.remove()

    # Get layers of model
    layers = models.list_layers(model)

    for layer in layers:
        if hasattr(layer, "saved_input"):
            del layer.saved_input


def save_backward_hook(module, grad_input, grad_output):
    """ """
    # Save module input
    module.saved_grad = module.grad.detach().cpu().numpy()

    # None as return value
    return None
