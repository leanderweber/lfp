import torch
from lxt import core as lcore
from lxt import functional as lfunctional
from lxt import rules as lrules
from lxt.modules import INIT_MODULE_MAPPING
from torch import nn
from zennit import types as ztypes

from ..model import activations
from ..model.custom_resnet import Sum


class ParameterizableComposite(lcore.Composite):
    """
    Allows for more parameterization of Rules
    """

    def __init__(self, layer_map, canonizers=[], zennit_composite=None) -> None:
        super(ParameterizableComposite, self).__init__(layer_map, canonizers, zennit_composite)
        self.original_inplace_states = []

    def _canonize_inplace_activations(self, parent: nn.Module):
        """
        Recursive function to iterate through the children of a module and set activation inplace argument to False.
        """

        for name, child in parent.named_children():
            if hasattr(child, "inplace"):
                self.original_inplace_states.append((child, child.inplace))
                child.inplace = False
            self._canonize_inplace_activations(child)

    def register(
        self,
        parent: nn.Module,
        dummy_inputs: dict = None,
        tracer=lcore.HFTracer,
        verbose=False,
        no_grad=True,
    ) -> None:
        # Activation "Canonization": remove inplace ops from activations, they don't work with LFP
        self._canonize_inplace_activations(parent)

        super(ParameterizableComposite, self).register(parent, dummy_inputs, tracer, verbose, no_grad)

    def remove(self):
        for module, inplace in self.original_inplace_states:
            module.inplace = inplace
        self.original_inplace_states = []
        super(ParameterizableComposite, self).remove()

    def _attach_module_rule(self, child, parent, name, rule_dict):
        """
        Attach a rule to the child module if it either 1. has a name equal to 'layer_type' or 2. is an instance of the
        'layer_type' (see rule_dict).
        In this case, if the rule is a subclass of WrapModule, the module is wrapped with the rule and attached to
        the parent as an attribute.
        If the rule is a torch.nn.Module, the module is directly replaced with the rule by copying the parameters and
        then attached to the parent as an attribute.
        """

        for layer_type, rule in rule_dict.items():
            # check if the layer_type is a string and or if the layer_type is a class and the child is an instance of it
            if (isinstance(layer_type, str) and layer_type == child) or isinstance(child, layer_type):
                if isinstance(rule, type) and issubclass(rule, lrules.WrapModule):
                    # replace module with LXT.rules.WrapModule and attach it to parent as attribute
                    xai_module = rule(child)
                elif not isinstance(rule, type) and isinstance(rule, RuleGenerator):
                    # RuleGenerator can be parameterized more
                    xai_module = rule(child)
                elif isinstance(rule, type) and issubclass(rule, nn.Module):
                    # replace module with LXT.module and attach it to parent as attribute
                    # INIT_MODULE_MAPPING contains the correct function for initializing and copying the parameters
                    # and buffers
                    xai_module = INIT_MODULE_MAPPING[rule](child, rule)
                    child = xai_module
                else:
                    raise ValueError(f"Rule {rule} must be a subclass of WrapModule or a torch.nn.Module")

                setattr(parent, name, xai_module)

                # save original module to revert the composite in self.remove()
                self.original_modules.append((parent, name, child))

                return child

        # could not find a rule for the module

        return child


class RuleGenerator:
    def __init__(self, rule, **kwargs):
        self.rule = rule
        self.rule_kwargs = kwargs

    def __call__(self, module):
        return self.rule(module, **self.rule_kwargs)


class LFPEpsilon(lrules.EpsilonRule):
    """
    LFP Epsilon Rule
    """

    def __init__(self, module, epsilon=1e-6, norm_backward=False, inplace=True):
        super(LFPEpsilon, self).__init__(module, epsilon)
        self.norm_backward = norm_backward
        self.inplace = inplace

    def forward(self, *inputs):
        return epsilon_lfp_fn.apply(self.module, self.epsilon, self.norm_backward, self.inplace, *inputs)


class epsilon_lfp_fn(lrules.epsilon_lrp_fn):
    """
    LFP Epsilon Rule
    """

    @staticmethod
    def forward(ctx, fn, epsilon, norm_backward, inplace, *inputs):
        # create boolean mask for inputs requiring gradients
        requires_grads = [True if inp.requires_grad else False for inp in inputs]
        if sum(requires_grads) == 0:
            # no gradients to compute or gradient checkpointing is used
            return fn(*inputs)

        # detach inputs to avoid overwriting gradients if same input is used as multiple arguments
        # (like in self-attention)
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        # get parameters to store for backward. Here, we want to accumulate reward, so we do not detach
        params = [param for _, param in fn.named_parameters(recurse=False)]

        for param in params:
            param.requires_grad_()

        with torch.enable_grad():
            outputs = fn(*inputs)

        ctx.epsilon, ctx.norm_backward, ctx.requires_grads, ctx.inplace = (
            epsilon,
            norm_backward,
            requires_grads,
            inplace,
        )
        # save only inputs requiring gradients
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, *params, outputs)

        ctx.n_inputs, ctx.n_params = len(inputs), len(params)
        ctx.fn = fn

        return outputs.detach()

    @staticmethod
    def backward(ctx, *incoming_reward):
        if ctx.norm_backward:
            if isinstance(incoming_reward, tuple):
                incoming_reward_new = []
                for g in incoming_reward:
                    if g is not None:
                        incoming_reward_new.append(
                            g
                            / torch.where(
                                g.abs().max() > 0,
                                g.abs().max(),
                                torch.ones_like(g.abs().max()),
                            )
                        )
                    else:
                        incoming_reward_new.append(None)
                incoming_reward = tuple(incoming_reward_new)
            else:
                if incoming_reward is not None:
                    incoming_reward = incoming_reward / torch.where(
                        incoming_reward.abs().max() > 0,
                        incoming_reward.abs().max(),
                        torch.ones_like(incoming_reward.abs().max()),
                    )
                else:
                    incoming_reward = None

        # if isinstance(ctx.fn, Sum):
        #     print([i.abs().max() for i in incoming_reward])

        outputs = ctx.saved_tensors[-1]
        inputs = ctx.saved_tensors[: ctx.n_inputs]
        params = ctx.saved_tensors[ctx.n_inputs : ctx.n_inputs + ctx.n_params]

        # print("OUTPUTS", outputs)
        # print("RELEVANCE", incoming_reward[0])

        normed_reward = incoming_reward[0] / lfunctional._stabilize(outputs, ctx.epsilon, inplace=False)

        # compute param reward (used to update parameters)
        for param in params:
            if not isinstance(param, tuple):
                param = (param,)  # noqa: PLW2901
            param_grads = torch.autograd.grad(outputs, param, normed_reward, retain_graph=True)
            if ctx.inplace:
                param_reward = tuple(param_grads[i].mul_(param[i].abs()) for i in range(len(param)))
            else:
                param_reward = tuple(param_grads[i] * param[i].abs() for i in range(len(param)))
            for i in range(len(param)):
                param[i].feedback = param_reward[i]
                # print(param[i].feedback.abs().max())

        # compute input reward (= outgoing reward to propagate)
        input_grads = torch.autograd.grad(outputs, inputs, normed_reward, retain_graph=False)

        if ctx.inplace:
            outgoing_reward = tuple(
                input_grads[i].mul_(inputs[i]) if ctx.requires_grads[i] else None
                for i in range(len(ctx.requires_grads))
            )
        else:
            outgoing_reward = tuple(
                input_grads[i] * inputs[i] if ctx.requires_grads[i] else None for i in range(len(ctx.requires_grads))
            )

        # return relevance at requires_grad indices else None
        return (None, None, None, None) + outgoing_reward


class LFPEpsilonComposite(ParameterizableComposite):
    def __init__(self, norm_backward=False, epsilon=1e-6):
        layer_map = {
            ztypes.Activation: lrules.IdentityRule,
            activations.Step: lrules.IdentityRule,
            Sum: RuleGenerator(LFPEpsilon, epsilon=epsilon, norm_backward=False, inplace=False),
            ztypes.AvgPool: RuleGenerator(LFPEpsilon, epsilon=epsilon, norm_backward=False),
            ztypes.Linear: RuleGenerator(LFPEpsilon, epsilon=epsilon, norm_backward=norm_backward),
            ztypes.BatchNorm: RuleGenerator(LFPEpsilon, epsilon=epsilon, norm_backward=norm_backward),
        }

        super().__init__(layer_map=layer_map)
