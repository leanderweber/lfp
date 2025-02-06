import torch

from . import reward_functions


class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self, *args, lower_bound=0.0, higher_bound=1.0, logit_sign_only=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.logit_sign_only = logit_sign_only

    def tensor_backward_hook(self, grad):
        # This completely overwrites the backward pass with the correct derivative
        # eye = torch.eye(self.stored_softmax.size()[1], device=self.stored_softmax.device)
        # one_hot = eye[self.stored_target]
        # retval = (self.stored_softmax-one_hot)
        retval = grad

        if self.logit_sign_only:
            if isinstance(retval, tuple):
                g_in = retval[0]
            else:
                g_in = retval
            retval = (g_in / self.stored_input.abs(),)

        retval = torch.where(
            self.stored_softmax > self.higher_bound,
            0.0,
            retval[0] if isinstance(retval, tuple) else retval,
        )
        retval = torch.where(
            self.stored_softmax < self.lower_bound,
            0.0,
            retval[0] if isinstance(retval, tuple) else retval,
        )

        return retval

    def forward(self, inp, target):
        # Store input for backward hook
        self.stored_input = inp
        self.stored_target = target
        if inp.requires_grad:
            inp.register_hook(self.tensor_backward_hook)

        softmax = torch.nn.functional.softmax(inp, dim=1)
        self.stored_softmax = softmax

        # Compute log_softmax (approximation) of bounded softmax
        log_softmax = torch.nn.functional.log_softmax(inp, dim=1)
        regularized_log_softmax = log_softmax
        regularized_log_softmax = torch.where(softmax > self.higher_bound, 0.0, regularized_log_softmax)
        regularized_log_softmax = torch.where(
            softmax < self.lower_bound, -99999.99999, regularized_log_softmax
        )  # "Correct" would be -np.inf for the bound; we approximate with very small number

        self.res = torch.nn.functional.nll_loss(regularized_log_softmax, target)

        return self.res


class SigmoidBCELossWrapper(torch.nn.BCELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, labels):
        logits = torch.sigmoid(logits)
        labels = labels.view_as(logits).float()
        return super().forward(logits, labels)


class MaximizeSingleNeuron(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, labels):
        # labels = torch.where(labels == 0, -1.0, 1.0)
        return -torch.sigmoid(logits)


class MinimizeSingleNeuron(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, labels):
        # labels = torch.where(labels == 0, -1.0, 1.0)
        return torch.sigmoid(logits)


REWARD_MAP = {
    "correct-class": reward_functions.SigmoidLossReward,  # Some older scripts use this
    "binarysigmoidlossreward": reward_functions.BinarySigmoidLossReward,
    "maximizesingleneuronreward": reward_functions.MaximizeSingleNeuron,
    "minimizesingleneuronreward": reward_functions.MinimizeSingleNeuron,
    "sigmoidlossreward": reward_functions.SigmoidLossReward,
    "softmaxlossreward": reward_functions.SoftmaxLossReward,
    "misclassificationreward": reward_functions.MisclassificationReward,
    "correctclassification": reward_functions.CorrectclassificationReward,
    "boundedsoftmaxreward": reward_functions.BoundedSoftmaxReward,
}

LOSS_MAP = {
    "ce-loss": torch.nn.CrossEntropyLoss,
    "custom-ce-loss": CustomCrossEntropyLoss,
    "bce-loss": SigmoidBCELossWrapper,
    "maximizesingleneuronloss": MaximizeSingleNeuron,
    "minimizesingleneuronloss": MinimizeSingleNeuron,
}


def get_reward(reward_name, device, *args, **kwargs):
    """
    Gets the correct reward function
    """

    # Check if model_name is supported
    if reward_name not in REWARD_MAP and reward_name not in LOSS_MAP:
        raise ValueError("Reward '{}' is not supported.".format(reward_name))

    # Build reward
    if reward_name in REWARD_MAP:
        reward_func = REWARD_MAP[reward_name](device=device, **kwargs)
    elif reward_name in LOSS_MAP:
        if reward_name == "custom-ce-loss":
            reward_func = LOSS_MAP[reward_name](**kwargs)
        else:
            reward_func = LOSS_MAP[reward_name]()

    # Return reward on correct device
    return reward_func
