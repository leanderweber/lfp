import torch


class MaximizeSingleNeuron:
    def __init__(self, device, **kwargs):
        """
        Computes reward based on Correct Class
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computation
        :param logits:
        :param labels:
        :return:
        """

        reward = logits.sign() * (1.0 - torch.sigmoid(logits))

        return reward


class MinimizeSingleNeuron:
    def __init__(self, device, **kwargs):
        """
        Computes reward based on Correct Class
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computation
        :param logits:
        :param labels:
        :return:
        """

        reward = logits.sign() * -(1.0 - torch.sigmoid(logits))

        return reward


class BinarySigmoidLossReward:
    def __init__(self, device, **kwargs):
        """
        Computes reward based on Correct Class
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computation
        :param logits:
        :param labels:
        :return:
        """

        reward = logits * (labels.view_as(logits) - torch.sigmoid(logits))

        return reward


class SigmoidLossReward:
    def __init__(self, device, **kwargs):
        """
        Computes reward based on Correct Class
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computation
        :param logits:
        :param labels:
        :return:
        """

        # Prepare one-hot labels
        eye = torch.eye(logits.size()[1], device=self.device)
        one_hot = eye[labels]

        reward = logits * (one_hot - torch.sigmoid(logits))

        return reward


class SoftmaxLossReward:
    def __init__(self, device, **kwargs):
        """
        Computes reward based on Correct Class
        """
        self.device = device

    def __call__(self, logits, labels):
        """
        Computation
        :param logits:
        :param labels:
        :return:
        """

        # Prepare one-hot labels
        eye = torch.eye(logits.size()[1], device=self.device)
        one_hot = eye[labels]

        # Compute reward
        reward = logits * (one_hot - torch.nn.functional.softmax(logits, dim=1))

        return reward


class BoundedSoftmaxReward:
    def __init__(self, device, lower_bound=0.0, higher_bound=1.0, logit_sign_only=False, **kwargs):
        """
        Computes reward based on Correct Class
        """
        self.device = device
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.logit_sign_only = logit_sign_only

    def __call__(self, logits, labels):
        """
        Computation
        :param logits:
        :param labels:
        :return:
        """

        eye = torch.eye(logits.size()[1], device=self.device)
        one_hot = eye[labels]

        regularized_softmax = torch.where(
            torch.nn.functional.softmax(logits, dim=1) > self.higher_bound,
            1.0,
            torch.nn.functional.softmax(logits, dim=1),
        )
        regularized_softmax = torch.where(regularized_softmax < self.lower_bound, 0.0, regularized_softmax)

        # Compute reward
        if self.logit_sign_only:
            reward = logits.sign() * (one_hot - regularized_softmax)
        else:
            reward = logits * (one_hot - regularized_softmax)

        return reward


class CorrectclassificationReward:
    def __init__(self, device, **kwargs):
        """
        Computes reward based on Correct Class
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computation
        :param logits:
        :param labels:
        :return:
        """

        # Prepare one-hot labels
        eye = torch.eye(logits.size()[1], device=self.device)
        one_hot = eye[labels]

        # Set all misclassifications to -1, everything else to 0
        # reward = torch.where(torch.stack([logits[l] > logits[l][label] for l, label in enumerate(labels)]), -1.0, 0.0)

        reward = torch.zeros_like(logits)

        # Set all correct classifications to 1
        for l, label in enumerate(labels):
            if logits[l].amax() == logits[l][label]:
                reward[l][label] = 1  # 1-torch.softmax(logits[l], dim=0)[label]

        # Correct Sign
        reward *= logits.sign()
        return reward


class MisclassificationReward:
    def __init__(self, device, **kwargs):
        """
        Computes reward based on Correct Class
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computation
        :param logits:
        :param labels:
        :return:
        """

        # Prepare one-hot labels
        eye = torch.eye(logits.size()[1], device=self.device)
        one_hot = eye[labels]

        reward = (
            torch.where(
                torch.stack([logits[l] > logits[l][label] for l, label in enumerate(labels)]),
                -1,
                0,
            )
            * logits.sign()
        )
        return reward
