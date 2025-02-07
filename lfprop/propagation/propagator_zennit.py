import copy
from contextlib import contextmanager

import torch
from zennit import composites
from zennit import core as zcore
from zennit import types as ztypes

from . import LAYER_MAP_BASE


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


def save_input_hook(module, inp):
    """
    Simple pytorch forward hook that writes a module's input into the module.saved_input parameter
    """
    # Save module input
    module.saved_input = inp[0]
    module.saved_input.requires_grad_()
    module.saved_input.retain_grad()

    # None as return value
    return None


@contextmanager
def mod_param_storage_context(module):
    """
    Enables Storage of any parameters modified within this context
    """
    try:
        if not hasattr(module, "stored_modparams"):
            module.stored_modparams = {}
        yield module
    finally:
        del module.stored_modparams


@contextmanager
def mod_params(module, modifier, param_keys=None, require_params=True):
    """
    Modifies Parameters temporarily
    """
    try:
        stored_params = {}
        if param_keys is None:
            param_keys = [name for name, _ in module.named_parameters(recurse=False)]

        missing = [key for key in param_keys if not hasattr(module, key)]
        if require_params and missing:
            raise RuntimeError("Module {} requires missing parameters: '{}'".format(module, "', '".join(missing)))

        for key in param_keys:
            if key not in missing:
                param = getattr(module, key)
                if param is not None:
                    stored_params[key] = param
                    setattr(
                        module,
                        key,
                        torch.nn.Parameter(modifier(copy.deepcopy(param.data), key)),
                    )

                    # Store modded params
                    if hasattr(module, "stored_modparams"):
                        if key not in module.stored_modparams.keys():
                            module.stored_modparams[key] = []
                        getattr(module, key).requires_grad_()
                        getattr(module, key).retain_grad()
                        module.stored_modparams[key].append(getattr(module, key))

        yield module
    finally:
        for key, value in stored_params.items():
            setattr(module, key, value)


class SpecialFirstNamedLayerMapComposite(composites.NameLayerMapComposite):
    """A Composite for which hooks are specified by a mapping from module types to hooks.

    Parameters
    ----------
    layer_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook.
    name_map: `list[tuple[tuple[str, ...], Hook]]`
        A mapping as a list of tuples, with a tuple of applicable module names and a Hook.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`.
    """

    def __init__(self, layer_map, name_map, first_map, canonizers=None):
        self.first_map = first_map
        super().__init__(name_map, layer_map, canonizers)

    # pylint: disable=unused-argument
    def mapping(self, ctx, name, module):
        """Get the appropriate hook given a mapping from module names to hooks.

        Parameters
        ----------
        ctx: dict
            A context dictionary to keep track of previously registered hooks.
        name: str
            Name of the module.
        module: obj:`torch.nn.Module`
            Instance of the module to find a hook for.

        Returns
        -------
        obj:`Hook` or None
            The hook found with the module type in the given name map, or None if no applicable hook was found.
        """
        # First Layer Map
        if not ctx.get("first_layer_visited", False):
            for types, hook in self.first_map:
                if isinstance(module, types):
                    ctx["first_layer_visited"] = True
                    return hook

        # Name and Layer Map
        return super().mapping(ctx, name, module)


class LFPHook(zcore.Hook):
    """
    Updates Weights of a model by propagating a reward backwards, like LRP propagates relevance
    """

    def __init__(
        self,
        norm_backward,
        input_modifiers=None,
        param_modifiers=None,
        output_modifiers=None,
        gradient_mapper=None,
        reducer=None,
        param_keys=None,
        require_params=True,
        is_bn=False,
    ):
        super().__init__()

        self.norm_backward = norm_backward
        self.is_bn = is_bn

        modifiers = {
            "in": input_modifiers,
            "param": param_modifiers,
            "out": output_modifiers,
        }
        supplied = {key for key, val in modifiers.items() if val is not None}
        num_mods = len(modifiers[next(iter(supplied))]) if supplied else 1
        modifiers.update({key: (self._default_modifier,) * num_mods for key in set(modifiers) - supplied})

        if gradient_mapper is None:
            gradient_mapper = self._default_gradient_mapper
        if reducer is None:
            reducer = self._default_reducer

        self.input_modifiers = modifiers["in"]
        self.param_modifiers = modifiers["param"]
        self.output_modifiers = modifiers["out"]
        self.gradient_mapper = gradient_mapper
        self.reducer = reducer

        self.param_keys = param_keys
        self.require_params = require_params

    def forward(self, module, inp, output):
        """Forward hook to save module in-/outputs."""
        self.stored_tensors["input"] = inp

    def backward(self, module, grad_input, grad_output):
        """
        Updates parameters of a layer
        :param layer:
        :return:
        """

        # Backwards Norm
        if self.norm_backward:
            if isinstance(grad_output, tuple):
                grad_output_new = []
                for g in grad_output:
                    if g is not None:
                        grad_output_new.append(
                            g
                            / torch.where(
                                g.abs().max() > 0,
                                g.abs().max(),
                                torch.ones_like(g.abs().max()),
                            )
                        )
                    else:
                        grad_output_new.append(None)
                grad_output = tuple(grad_output_new)
            else:
                if grad_output is not None:
                    grad_output = grad_output / torch.where(
                        grad_output.abs().max() > 0,
                        grad_output.abs().max(),
                        torch.ones_like(grad_output.abs().max()),
                    )
                else:
                    grad_output = None

        original_input = self.stored_tensors["input"][0].detach()
        param_kwargs = dict(param_keys=self.param_keys, require_params=self.require_params)
        inputs = []
        outputs = []

        with mod_param_storage_context(module) as param_storing_module:
            for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
                inp = in_mod(original_input).requires_grad_()
                with (
                    mod_params(param_storing_module, param_mod, **param_kwargs) as modified,
                    torch.autograd.enable_grad(),
                ):
                    output = modified.forward(inp)
                    output = out_mod(output)

                inputs.append(inp)
                outputs.append(output)

            # Input Relevance
            input_gradients = torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs=self.gradient_mapper(grad_output[0], outputs),
                retain_graph=True,
                create_graph=False,
            )
            input_relevance = self.reducer(inputs, input_gradients)

            # Parameter Relevance
            param_keys = [name for name, _ in param_storing_module.named_parameters(recurse=False)]
            for key in param_keys:
                param_datas = param_storing_module.stored_modparams[key]
                param_gradients = torch.autograd.grad(
                    outputs,
                    param_datas,
                    grad_outputs=self.gradient_mapper(grad_output[0], outputs),
                    retain_graph=True,
                    create_graph=False,
                )

                param_relevance = self.reducer(param_datas, param_gradients)

                # LFP
                param_reward = param_relevance
                param_reward *= getattr(module, key).data.sign()

                # Get param_reward mean
                param_reward /= grad_output[0].shape[0]
                # Set Parameter Grad
                getattr(module, key).feedback = param_reward

        return tuple(input_relevance if original.shape == input_relevance.shape else None for original in grad_input)

    def copy(self):
        """Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        """
        return LFPHook(
            self.norm_backward,
            self.input_modifiers,
            self.param_modifiers,
            self.output_modifiers,
            self.gradient_mapper,
            self.reducer,
            self.param_keys,
            self.require_params,
            self.is_bn,
        )

    @staticmethod
    def _default_modifier(obj, name=None):
        return obj

    @staticmethod
    def _default_gradient_mapper(out_grad, outputs):
        return tuple(out_grad / zcore.stabilize(output) for output in outputs)

    @staticmethod
    def _default_reducer(inputs, gradients):
        return sum(inp * gradient for inp, gradient in zip(inputs, gradients))


class LFPEpsilon(LFPHook):
    """Epsilon LFP rule.

    Parameters
    ----------
    epsilon: float, optional
        Stabilization parameter.
    """

    def __init__(self, norm_backward, epsilon=1e-6):
        super().__init__(
            norm_backward=norm_backward,
            input_modifiers=[lambda inp: inp],
            param_modifiers=[lambda param, _: param],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / zcore.stabilize(outputs[0], epsilon)),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class LFPEpsilonComposite(SpecialFirstNamedLayerMapComposite):
    def __init__(
        self,
        norm_backward=False,
        epsilon=1e-6,
        canonizers=None,
    ):
        layer_map = LAYER_MAP_BASE + [
            (
                ztypes.Linear,
                LFPEpsilon(norm_backward=norm_backward, epsilon=epsilon),
            ),
            (
                ztypes.BatchNorm,
                LFPEpsilon(norm_backward=norm_backward, epsilon=epsilon),
            ),
        ]
        name_map = []
        first_map = []

        super().__init__(
            layer_map=layer_map,
            name_map=name_map,
            first_map=first_map,
            canonizers=canonizers,
        )
