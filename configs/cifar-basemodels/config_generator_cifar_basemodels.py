import os

import yaml

config_dir = "configs/cifar-basemodels/"

# os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

base_config = {
    "savepath": "/mnt/output",
    "batch_size": 128,
    "n_channels": 3,
    "momentum": 0.9,
    "clip_update_threshold": 3.0,
    "reward_name": "softmaxlossreward",
    "reward_kwargs": {},
    "loss_name": "ce-loss",
    "base_epochs": 50,
    "model_name": "cifar-vgglike",
    "activation": "relu",
    "batch_log": False,
    "wandb_key": "<wandb-key>",
    "disable_wandb": False,
    "wandb_project_name": "cifar-basemodels",
    "transfer_training": False,
    "verbose": False,
}

BASE_DATASET_NAMES = ["cifar10", "cifar100"]
BASE_LRS = [0.1]
PROPAGATOR_NAMES = ["lfp-epsilon", "vanilla-gradient"]
CLIP_UPDATES = [True]
NORM_BACKWARDS = [False]
WEIGHT_DECAYS = [0.0]
SCHEDULER_NAMES = ["onecyclelr"]
SEEDS = [7240, 5110, 5628]

for base_dataset_name in BASE_DATASET_NAMES:
    for base_lr in BASE_LRS:
        for propagator_name in PROPAGATOR_NAMES:
            for clip_updates in CLIP_UPDATES:
                for norm_backward in NORM_BACKWARDS:
                    for weight_decay in WEIGHT_DECAYS:
                        for scheduler_name in SCHEDULER_NAMES:
                            for seed in SEEDS:
                                base_config["base_dataset_name"] = base_dataset_name
                                base_config["base_lr"] = base_lr
                                base_config["propagator_name"] = propagator_name
                                base_config["seed"] = seed

                                base_config["base_data_path"] = "/mnt/data/"
                                base_config["clip_updates"] = clip_updates
                                base_config["norm_backward"] = norm_backward
                                base_config["weight_decay"] = weight_decay
                                base_config["scheduler_name"] = scheduler_name

                                config_name = f"{base_config['base_dataset_name']}_{base_config['base_lr']}"
                                config_name += f"_{base_config['propagator_name']}_{base_config['norm_backward']}"
                                config_name += f"_{base_config['clip_updates']}_{base_config['weight_decay']}"
                                config_name += f"_{base_config['scheduler_name']}_{base_config['seed']}"

                                with open(
                                    f"{config_dir}/cluster/{config_name}_basemodel.yaml",
                                    "w",
                                ) as outfile:
                                    yaml.dump(base_config, outfile, default_flow_style=False)
