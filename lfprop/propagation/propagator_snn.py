"""
Propagator for training SNN with LFP
"""

from contextlib import contextmanager

import torch
from zennit import core as zcore

from lfprop.model import spiking_networks as models


def collect_leaves(module):
    """
    From Zennit.
    Generator function to collect all leaf modules of a module.
    """
    is_leaf = True

    children = module.children()
    for child in children:
        is_leaf = False
        for leaf in collect_leaves(child):
            yield leaf
    if is_leaf:
        yield module


@contextmanager
def mod_params(module, modifier, param_keys=None, require_params=True):
    """Context manager to temporarily modify parameter attributes (all by default) of a module.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        Module of which to modify parameters. If `requires_params` is `True`, it must have all elements given in
        `param_keys` as attributes (attributes are allowed to be `None`, in which case they are ignored).
    modifier: function
        A function used to modify parameter attributes. If `param_keys` is empty, this is not used.
    param_keys: list[str], optional
        A list of parameters that shall be modified. If `None` (default), all parameters are modified (which may be
        none). If `[]`, no parameters are modified and `modifier` is ignored.
    require_params: bool, optional
        Whether existence of `module`'s params is mandatory (True by default). If the attribute exists but is `None`,
        it is not considered missing, and the modifier is not applied.

    Raises
    ------
    RuntimeError
        If `require_params` is `True` and `module` is missing an attribute listed in `param_keys`.

    Yields
    ------
    module: obj:`torch.nn.Module`
        The `module` with appropriate parameters temporarily modified.
    """
    try:
        stored_tensors = {}
        if param_keys is None:
            param_keys = [name for name, _ in module.named_parameters(recurse=False)]

        missing = [key for key in param_keys if not hasattr(module, key)]
        if require_params and missing:
            raise RuntimeError("Module {} requires missing parameters: '{}'".format(module, "', '".join(missing)))

        for key in param_keys:
            if key not in missing:
                param = getattr(module, key)
                if param is not None:
                    stored_tensors[key] = param.data
                    param.data = modifier(param.data, key)
        yield module
    finally:
        for key, value in stored_tensors.items():
            getattr(module, key).data = value


class LRPRewardPropagator:
    """
    Updates Weights of a model by propagating a reward backwards, like LRP propagates relevance
    """

    def __init__(self, model, norm_backward):
        self.model = model
        self.norm_backward = norm_backward

    def update_snntorch_parameters(self, layer, iteration_feedback, iteration_idx):
        """
        Updates parameters of a snn layer tuple
        """

        layer_Xcur = layer[0].stored_x[::-1][iteration_idx]
        layer_W = layer[0].weight
        layer_WXcur = layer[0].stored_out[::-1][iteration_idx]
        layer_beta = layer[-1].beta
        layer_Uprev = layer[-1].stored_mem[::-1][iteration_idx + 1]
        layer_Ucur = layer[-1].stored_mem[::-1][iteration_idx]
        layer_threshold = layer[-1].threshold
        layer_reset = layer[-1].stored_reset[::-1][iteration_idx]
        layer_minmem = layer[-1].minmem if layer[-1].minmem is not None else 0

        layer_feedback = iteration_feedback.view(layer_Ucur.shape)
        if hasattr(layer[-1], "stored_feedback"):
            layer_feedback += layer[-1].stored_feedback

        if self.norm_backward:
            max_abs_fb_val = layer_feedback.abs().amax()
            if max_abs_fb_val > 1e-10:
                layer_feedback /= max_abs_fb_val
            else:
                print("WARN: norm-backward not applied due to div by 0 otherwise,")

        layer_param_feedback = layer_feedback

        layer_param_feedback /= zcore.stabilize(layer_Ucur + layer_reset * layer_threshold - layer_minmem, 1e-6).abs()

        for between_layer in list(layer)[1:-1][::-1]:
            layer_param_feedback = between_layer.backward_lfp(layer_param_feedback)

        layer_Uprev_feedback = (
            layer_beta
            * layer_Uprev
            / zcore.stabilize(layer_Ucur + layer_reset * layer_threshold - layer_minmem, 1e-6).abs()
            * layer_feedback
        ).detach()
        layer_W_feedback = (
            layer_W.data
            * torch.autograd.grad(
                layer_WXcur,
                layer_W,
                grad_outputs=layer_param_feedback,
                retain_graph=True,
            )[0]
        ).detach()

        if hasattr(layer[0], "bias") and layer[0].bias is not None:
            layer_bias_feedback = (
                layer[0].bias.data
                * torch.autograd.grad(
                    layer_WXcur,
                    layer[0].bias,
                    grad_outputs=layer_param_feedback,
                    retain_graph=True,
                )[0]
            ).detach()

        layer_Xcur_feedback = (
            layer_Xcur
            * torch.autograd.grad(
                layer_WXcur,
                layer_Xcur,
                grad_outputs=layer_param_feedback,
                retain_graph=False,
            )[0]
        ).detach()

        layer[-1].stored_feedback = layer_Uprev_feedback

        layer_W_feedback *= layer_W.data.sign()
        if hasattr(layer[0], "bias") and layer[0].bias is not None:
            layer_bias_feedback *= layer[0].bias.data.sign()

        layer_W_feedback /= iteration_feedback.shape[0]
        if hasattr(layer[0], "bias") and layer[0].bias is not None:
            layer_bias_feedback /= iteration_feedback.shape[0]

        if not hasattr(layer[0].weight, "accumulated_feedback"):
            layer[0].weight.accumulated_feedback = 0
        layer[0].weight.accumulated_feedback += layer_W_feedback.detach()

        if hasattr(layer[0], "bias") and layer[0].bias is not None:
            if not hasattr(layer[0].bias, "accumulated_feedback"):
                layer[0].bias.accumulated_feedback = 0
            layer[0].bias.accumulated_feedback += layer_bias_feedback.detach()

        if any([torch.isnan(t).sum() > 0 for t in [layer_Uprev_feedback, layer_W_feedback, layer_bias_feedback]]):
            raise ValueError(
                "LFP Backprop results contain nan values, probably due to exploding relevances. \n"
                "If this persists, try varying the following hyperparameters: \n"
                " -> set norm_backward to True \n"
                " -> set clip_update to True \n"
                " -> lower the learning rate \n"
                " -> use a different optimizer"
            )

        return layer_Xcur_feedback

    def propagate(self, iteration_feedback, iteration_idx):
        """
        Propagates reward backwards
        :param iteration_feedback: feedback evaluating the model prediction
        :param iteration_idx: index for iterating over SNN prediction timesteps in REVERSE
        :return: -
        """

        layers = models.list_snn_layers(self.model)

        for lay, layer in enumerate(reversed(layers)):
            if isinstance(layer, tuple):
                iteration_feedback = self.update_snntorch_parameters(layer, iteration_feedback, iteration_idx)
            else:
                raise ValueError("This model seems to not be a SNN")

    def reset(self):
        """
        Resets all feedbacks
        """
        layers = models.list_snn_layers(self.model)
        for lay, layer in enumerate(reversed(layers)):
            for module in layer:
                if hasattr(module, "stored_feedback"):
                    del module.stored_feedback
                if hasattr(module, "weight"):
                    if hasattr(module.weight, "accumulated_feedback"):
                        del module.weight.accumulated_feedback
                if hasattr(module, "bias"):
                    if hasattr(module.bias, "accumulated_feedback"):
                        del module.bias.accumulated_feedback

        for layer in layers:
            for lay in layer:
                assert not hasattr(lay, "stored_feedback"), f"{lay} {layer}"


class ZplusMinusPropagator(LRPRewardPropagator):
    def __init__(self, model, norm_backward):
        super().__init__(model, norm_backward)

    def update_snntorch_parameters(self, layer, iteration_feedback, iteration_idx):
        """
        Updates parameters of a snn layer tuple
        """

        layer_Xcur = layer[0].stored_x[::-1][iteration_idx]
        layer_W = layer[0].weight
        layer[0].stored_out[::-1][iteration_idx]
        layer_beta = layer[-1].beta
        layer_Uprev = layer[-1].stored_mem[::-1][iteration_idx + 1]
        layer_Ucur = layer[-1].stored_mem[::-1][iteration_idx]
        layer[-1].threshold
        layer[-1].stored_reset[::-1][iteration_idx]
        layer[-1].minmem if layer[-1].minmem is not None else 0

        layer_feedback = iteration_feedback.view(layer_Ucur.shape)
        if hasattr(layer[-1], "stored_feedback"):
            layer_feedback += layer[-1].stored_feedback

        if self.norm_backward:
            max_abs_fb_val = layer_feedback.abs().amax()
            if max_abs_fb_val > 1e-10:
                layer_feedback /= max_abs_fb_val
            else:
                print("WARN: norm-backward not applied due to div by 0 otherwise,")

        with (
            mod_params(layer[0], lambda p, _: p.clip(min=0)) as modified,
            torch.autograd.enable_grad(),
        ):
            layer_WXcur_pos = modified(layer_Xcur)
            layer_WXcur_pos_prop = layer_WXcur_pos
            for between_layer in list(layer)[1:-1]:
                layer_WXcur_pos_prop = between_layer(layer_WXcur_pos_prop)
        with (
            mod_params(layer[0], lambda p, _: p.clip(max=0)) as modified,
            torch.autograd.enable_grad(),
        ):
            layer_WXcur_neg = modified(layer_Xcur)
            layer_WXcur_neg_prop = layer_WXcur_neg
            for between_layer in list(layer)[1:-1]:
                layer_WXcur_neg_prop = between_layer(layer_WXcur_neg_prop)

        eps = 1e-6
        layer_Uprev_pos = layer_Uprev.clip(min=0) if not isinstance(layer_Uprev, float) else min(0.0, layer_Uprev)
        layer_Uprev_neg = layer_Uprev.clip(max=0) if not isinstance(layer_Uprev, float) else max(0.0, layer_Uprev)
        denominator_pos = layer_WXcur_pos_prop + layer_beta * layer_Uprev_pos
        denominator_pos = torch.where(
            denominator_pos == 0,
            torch.ones_like(denominator_pos) * eps,
            denominator_pos,
        )
        denominator_neg = layer_WXcur_neg_prop + layer_beta * layer_Uprev_neg
        denominator_neg = torch.where(
            denominator_neg == 0,
            -torch.ones_like(denominator_pos) * eps,
            denominator_neg,
        )

        pos_part = layer_WXcur_pos_prop.abs() / (layer_WXcur_neg_prop.abs() + layer_WXcur_pos_prop.abs())
        neg_part = layer_WXcur_neg_prop.abs() / (layer_WXcur_neg_prop.abs() + layer_WXcur_pos_prop.abs())
        layer_param_feedback_pos = pos_part * layer_feedback / denominator_pos
        layer_param_feedback_neg = neg_part * layer_feedback / denominator_neg

        layer_Uprev_feedback = 0
        layer_Uprev_feedback += (layer_beta * layer_Uprev_pos * layer_param_feedback_pos).detach()
        layer_Uprev_feedback -= (layer_beta * layer_Uprev_neg * layer_param_feedback_neg).detach()

        for between_layer in list(layer)[1:-1][::-1]:
            layer_param_feedback_pos = between_layer.backward_lfp(layer_param_feedback_pos)
            layer_param_feedback_neg = between_layer.backward_lfp(layer_param_feedback_neg)

        layer_W_feedback = 0
        layer_bias_feedback = 0
        param_keys = [name for name, _ in layer[0].named_parameters(recurse=False)]
        for key in param_keys:
            with (
                mod_params(layer[0], lambda p, _: p.clip(min=0)) as modified,
                torch.autograd.enable_grad(),
            ):
                if "weight" in key and getattr(modified, key) is not None:
                    param = getattr(modified, key)
                    layer_W_feedback += (
                        param.data
                        * torch.autograd.grad(
                            layer_WXcur_pos,
                            param,
                            grad_outputs=layer_param_feedback_pos,
                            retain_graph=True,
                        )[0]
                    ).detach()
                elif "bias" in key and getattr(modified, key) is not None:
                    param = getattr(modified, key)
                    layer_bias_feedback += (
                        param.data
                        * torch.autograd.grad(
                            layer_WXcur_pos,
                            param,
                            grad_outputs=layer_param_feedback_pos,
                            retain_graph=True,
                        )[0]
                    ).detach()

            with (
                mod_params(layer[0], lambda p, _: p.clip(max=0)) as modified,
                torch.autograd.enable_grad(),
            ):
                if "weight" in key and getattr(modified, key) is not None:
                    param = getattr(modified, key)
                    layer_W_feedback -= (
                        param.data
                        * torch.autograd.grad(
                            layer_WXcur_neg,
                            param,
                            grad_outputs=layer_param_feedback_neg,
                            retain_graph=True,
                        )[0]
                    ).detach()
                elif "bias" in key and getattr(modified, key) is not None:
                    param = getattr(modified, key)
                    layer_bias_feedback -= (
                        param.data
                        * torch.autograd.grad(
                            layer_WXcur_neg,
                            param,
                            grad_outputs=layer_param_feedback_neg,
                            retain_graph=True,
                        )[0]
                    ).detach()

        layer_Xcur_feedback = 0
        layer_Xcur_feedback += (
            layer_Xcur
            * torch.autograd.grad(
                layer_WXcur_pos,
                layer_Xcur,
                grad_outputs=layer_param_feedback_pos,
                retain_graph=False,
            )[0]
        ).detach()
        layer_Xcur_feedback += (
            layer_Xcur
            * torch.autograd.grad(
                layer_WXcur_neg,
                layer_Xcur,
                grad_outputs=layer_param_feedback_neg,
                retain_graph=False,
            )[0]
        ).detach()

        layer[-1].stored_feedback = layer_Uprev_feedback

        layer_W_feedback *= layer_W.data.sign()
        if hasattr(layer[0], "bias") and layer[0].bias is not None:
            layer_bias_feedback *= layer[0].bias.data.sign()

        layer_W_feedback /= iteration_feedback.shape[0]
        if hasattr(layer[0], "bias") and layer[0].bias is not None:
            layer_bias_feedback /= iteration_feedback.shape[0]

        if not hasattr(layer[0].weight, "accumulated_feedback"):
            layer[0].weight.accumulated_feedback = 0
        layer[0].weight.accumulated_feedback += layer_W_feedback.detach()

        if hasattr(layer[0], "bias") and layer[0].bias is not None:
            if not hasattr(layer[0].bias, "accumulated_feedback"):
                layer[0].bias.accumulated_feedback = 0
            layer[0].bias.accumulated_feedback += layer_bias_feedback.detach()

        if any([torch.isnan(t).sum() > 0 for t in [layer_Uprev_feedback, layer_W_feedback, layer_bias_feedback]]):
            raise ValueError(
                "LFP Backprop results contain nan values, probably due to exploding relevances. \n"
                "If this persists, try varying the following hyperparameters: \n"
                " -> set norm_backward to True \n"
                " -> set clip_update to True \n"
                " -> lower the learning rate \n"
                " -> use a different optimizer"
            )

        return layer_Xcur_feedback
