try:
    import snntorch as snn
    from snntorch import utils as snnutils
except ImportError:
    print(
        "The SNN functionality of this package requires extra dependencies ",
        "which can be installed via pip install lfprop[snn] (or lfprop[full] for all dependencies).",
    )
    raise ImportError("snntorch required; reinstall lfprop with option `snn` (pip install lfprop[snn])")


import torch
from torch import nn as tnn

# Model definitions


class CustomLeaky(snn.Leaky):
    def __init__(self, *args, **kwargs):
        self.minmem = kwargs.pop("minmem", None)
        super().__init__(*args, **kwargs)
        self.fire = self.custom_fire
        self.mem_reset = self.custom_mem_reset

    def custom_fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""

        if self.state_quant:
            mem = self.state_quant(mem)

        spk = (mem > self.threshold).float()

        return spk

    def custom_mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        reset = (mem > self.threshold).float()

        return reset

    def forward(self, input_, mem=None):
        """
        Clips mem if desired
        """
        if self.minmem is None:
            return super().forward(input_, mem)
        else:
            if not self.init_hidden:
                spk, mem = super().forward(input_, mem)
                return spk, mem.clip(min=self.minmem)
            else:
                _ = super().forward(input_, mem)
                self.mem = self.mem.clip(min=self.minmem)
                if self.output:
                    return self.spk, self.mem
                else:
                    return self.mem


class CustomMaxPool2d(tnn.MaxPool2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_indices = True
        self.idx = None
        self.input_shape = None

    def forward(self, input):
        self.input_size = input.size()
        res, self.idx = super().forward(input)
        return res

    def backward_lfp(self, incoming_feedback):
        outgoing_feedback = torch.nn.functional.max_unpool2d(
            incoming_feedback,
            indices=self.idx,
            kernel_size=self.kernel_size,
            stride=self.stride,
            output_size=self.input_size,
        )
        return outgoing_feedback


class LifMLP(tnn.Module):
    """
    Simple MLP using Leaky-Integrate-And-Fire Neurons
    """

    def __init__(self, n_channels, n_outputs, beta, minmem, **kwargs):
        super().__init__()

        # Classifier
        self.classifier = tnn.Sequential(
            tnn.Linear(n_channels, 1000),
            CustomLeaky(beta=beta, init_hidden=True, minmem=minmem, **kwargs),
            tnn.Linear(1000, 1000),
            CustomLeaky(beta=beta, init_hidden=True, minmem=minmem, **kwargs),
            tnn.Linear(1000, n_outputs),
            CustomLeaky(beta=beta, init_hidden=True, output=True, minmem=minmem),
        )

        self.forward_handles = []

    def register_forward_hooks(self):
        """
        Registers forward hooks to save necessary stuff
        :return:
        """
        for layer in self.classifier.modules():
            if not isinstance(layer, snn.SpikingNeuron):
                self.forward_handles.append(layer.register_forward_pre_hook(save_input_hook))
                self.forward_handles.append(layer.register_forward_hook(save_output_hook))

    def remove_forward_hooks(self):
        """
        Removes forward hooks
        :return:
        """
        # Remove forward hooks
        for handle in self.forward_handles:
            handle.remove()

    def save_forward_states(self):
        for layer in self.classifier.modules():
            if isinstance(layer, snn.SpikingNeuron):
                layer.stored_mem.append(layer.mem.detach())
                layer.stored_reset.append(layer.reset.detach())

    def reset(self):
        # Reset states and store initial states
        for layer in self.classifier.modules():
            if isinstance(layer, snn.SpikingNeuron):
                layer.stored_mem = [0.0]
                layer.stored_reset = []
            else:
                layer.stored_x = []
                layer.stored_out = []

        # SNN Reset. Careful with this: Needs to have sequential model passed as this functions does not seem to iterate
        # through modules properly in all cases (e.g. do NOT pass self instead of self.classifier)
        snnutils.reset(self.classifier)

    def forward(self, x):
        """
        Forwards input through network
        """

        if self.training:
            self.register_forward_hooks()

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.training:
            self.save_forward_states()
            self.remove_forward_hooks()

        # Return output
        return x


class SmallLifMLP(LifMLP):
    def __init__(self, n_channels, n_outputs, beta, minmem, **kwargs):
        super().__init__(n_channels, n_outputs, beta, minmem, **kwargs)

        # Classifier
        self.classifier = tnn.Sequential(
            tnn.Linear(n_channels, 1000),
            CustomLeaky(beta=beta, init_hidden=True, minmem=minmem, **kwargs),
            # tnn.Linear(1000, 1000),
            # CustomLeaky(beta=beta, init_hidden=True, minmem=minmem, **kwargs),
            tnn.Linear(1000, n_outputs),
            CustomLeaky(beta=beta, init_hidden=True, output=True, minmem=minmem),
        )


class LifCNN(LifMLP):
    """
    Simple CNN using Leaky-Integrate-And-Fire Neurons
    """

    def __init__(self, n_channels, n_outputs, beta, minmem, **kwargs):
        super().__init__(n_channels, n_outputs, beta, minmem)

        # Classifier
        self.classifier = tnn.Sequential(
            tnn.Conv2d(n_channels, 12, 5),
            CustomMaxPool2d(2),
            CustomLeaky(beta=beta, init_hidden=True, minmem=minmem, **kwargs),
            tnn.Conv2d(12, 64, 5),
            CustomMaxPool2d(2),
            CustomLeaky(beta=beta, init_hidden=True, minmem=minmem, **kwargs),
            tnn.Flatten(),
            tnn.Linear(64 * 4 * 4, n_outputs),
            CustomLeaky(beta=beta, init_hidden=True, output=True, minmem=minmem),
        )

    def forward(self, x):
        """
        Forwards input through network
        """

        if self.training:
            self.register_forward_hooks()

        x = self.classifier(x)

        if self.training:
            self.save_forward_states()
            self.remove_forward_hooks()

        # Return output
        return x


class GradLifMLP(tnn.Module):
    """
    Simple MLP using Leaky-Integrate-And-Fire Neurons
    """

    def __init__(self, n_channels, n_outputs, beta, spike_grad, **kwargs):
        super().__init__()

        # Classifier
        self.classifier = tnn.Sequential(
            tnn.Linear(n_channels, 1000),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, **kwargs),
            tnn.Linear(1000, 1000),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, **kwargs),
            tnn.Linear(1000, n_outputs),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True),
        )

        self.forward_handles = []

    def reset(self):
        snnutils.reset(self.classifier)

    def forward(self, x):
        """
        Forwards input through network
        """

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # Return output
        return x


class GradLifCNN(GradLifMLP):
    """
    Simple CNN using Leaky-Integrate-And-Fire Neurons
    """

    def __init__(self, n_channels, n_outputs, beta, spike_grad, **kwargs):
        super().__init__(n_channels, n_outputs, beta, spike_grad)

        # Classifier
        self.classifier = tnn.Sequential(
            tnn.Conv2d(n_channels, 12, 5),
            tnn.MaxPool2d(2),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, **kwargs),
            tnn.Conv2d(12, 64, 5),
            tnn.MaxPool2d(2),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, **kwargs),
            tnn.Flatten(),
            tnn.Linear(64 * 4 * 4, n_outputs),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True),
        )

    def forward(self, x):
        """
        Forwards input through network
        """

        x = self.classifier(x)

        # Return output
        return x


# Helper functions

MODEL_MAP = {
    "lifmlp": LifMLP,
    "smalllifmlp": SmallLifMLP,
    "lifcnn": LifCNN,
    "gradlifmlp": GradLifMLP,
    "gradlifcnn": GradLifCNN,
}

BASE_LAYERS = [torch.nn.Linear, torch.nn.Conv2d]

EXCLUDED_MODULE_TYPES = [LifMLP, LifCNN]


def init_uniform(m):
    if isinstance(m, tnn.Linear):
        torch.nn.init.uniform_(m.weight, 0.0, 1.0)


def save_input_hook(module, input):
    """
    Simple pytorch forward hook that saves input
    """
    if isinstance(input, tuple):
        tmp_in = input[0]
    else:
        tmp_in = input
    tmp_in.requires_grad_()
    tmp_in.retain_grad()
    module.stored_x += [tmp_in]


def save_output_hook(module, input, output):
    """
    Simple pytorch forward hook that saves output
    """
    if isinstance(output, tuple):
        tmp_out = output[0]
    else:
        tmp_out = output
    tmp_out.requires_grad_()
    tmp_out.retain_grad()
    module.stored_out += [tmp_out]

    # None as return value
    return None


def get_model(model_name, n_channels, n_outputs, device, **kwargs):
    """
    Gets the correct model
    """

    # Check if model_name is supported
    if model_name not in MODEL_MAP:
        raise ValueError("Model '{}' is not supported.".format(model_name))

    # Build model
    if model_name in MODEL_MAP:
        model = MODEL_MAP[model_name](
            n_channels=n_channels,
            n_outputs=n_outputs,
            **kwargs,
        )

    model.reset()  # necessary for SNNs (see snntorch documentation)
    # Return model on correct device
    return model.to(device)


def list_layers(model):
    """
    List module layers
    """

    # Exclude specific types of modules
    layers = [module for module in model.modules() if type(module) not in [torch.nn.Sequential] + EXCLUDED_MODULE_TYPES]

    return layers


def list_snn_layers(model):
    """
    List module layers for SNNs. I.e., layers are returned as tuples of some torch layer and the following snntorch activation layer
    """

    # Exclude specific types of modules
    layers = [module for module in model.modules() if type(module) not in [torch.nn.Sequential] + EXCLUDED_MODULE_TYPES]

    # Go through layers, grouping each snntorch layer with the preceding torch layer
    rev_layers = layers[::-1]
    snnlayers = []
    for l, layer in enumerate(rev_layers):
        if isinstance(layer, snn.SpikingNeuron):
            next_base_idx = l
            base_layer = rev_layers[next_base_idx]
            tmp = [rev_layers[l]]
            while not any([isinstance(base_layer, bl) for bl in BASE_LAYERS]):
                next_base_idx += 1
                base_layer = rev_layers[next_base_idx]
                tmp.append(base_layer)

            snnlayers.append(tuple(tmp[::-1]))

    return snnlayers[::-1]


def clip_gradients(model, clip_update, clip_update_threshold=0.06):
    """
    Clips gradients of model parameters (unit-wise) using frobenius norms
    """

    if clip_update:
        for layer in list_layers(model):
            # Get Parameters
            param_keys = [name for name, _ in layer.named_parameters(recurse=False)]

            sum = 0.0
            frob_norm_p = 0.0
            for key in param_keys:
                val = getattr(layer, key).data
                if len(val.shape) == 1:
                    val = val.unsqueeze(1)
                sum += (val**2).sum(dim=list(range(len(val.shape)))[1:])
            if len(param_keys) != 0:
                frob_norm_p = sum.sqrt()

            for key in param_keys:
                param = getattr(layer, key)
                param_grads = param.grad
                if param_grads is not None:
                    param_grads_shape = param_grads.shape
                    if len(param_grads_shape) == 1:
                        param_grads = param_grads.unsqueeze(1)
                    frob_norm_u = (param_grads**2).sum(dim=list(range(len(param_grads.shape)))[1:]).sqrt()
                    frob_norm_p_norm = torch.amax(
                        torch.stack([frob_norm_p, torch.ones_like(frob_norm_p) * 1e-6]),
                        dim=0,
                    )
                    cond = frob_norm_u / frob_norm_p_norm > clip_update_threshold
                    repl = (
                        clip_update_threshold
                        * frob_norm_p_norm
                        / torch.where(
                            frob_norm_u == 0,
                            torch.ones_like(frob_norm_u) * 1e-6,
                            frob_norm_u,
                        )
                    )

                    repl = param_grads * repl.view(
                        param_grads.shape[0],
                        *[1 for _ in range(len(param_grads.shape))[1:]],
                    ).repeat(1, *param_grads.shape[1:])

                    param_grads = torch.where(
                        cond.view(
                            param_grads.shape[0],
                            *[1 for _ in range(len(param_grads.shape))[1:]],
                        ).repeat(1, *param_grads.shape[1:]),
                        repl,
                        param_grads,
                    )
                    param_grads = param_grads.view(param_grads_shape)
                    param.grad = param_grads
