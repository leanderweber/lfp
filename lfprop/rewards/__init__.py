import torch

from lfprop.rewards.reward_functions import SnnCorrectClassRewardSpikesRateCoded

REWARD_MAP = {
    "correct-class-spikes-ratecoded": SnnCorrectClassRewardSpikesRateCoded,
}

LOSS_MAP = {
    "ce-loss": torch.nn.CrossEntropyLoss,
    "bce-loss": torch.nn.BCEWithLogitsLoss,
}


def get_reward(reward_name, device):
    """
    Gets the correct reward function
    """

    # Check if model_name is supported
    if reward_name not in REWARD_MAP and reward_name not in LOSS_MAP:
        raise ValueError("Reward '{}' is not supported.".format(reward_name))

    # Build reward
    if reward_name in REWARD_MAP:
        reward_func = REWARD_MAP[reward_name](
            device=device,
        )
    elif reward_name in LOSS_MAP:
        reward_func = LOSS_MAP[reward_name]()

    # Return reward on correct device
    return reward_func
