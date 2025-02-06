import os

import yaml

config_dir = "configs/imagenet-transfer/"

# os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

base_config = {
    "savepath": "/mnt/output",
    "base_data_path": "/mnt/data/imagenet",
    "base_dataset_name": "imagenet",
    "batch_size": 32,
    "n_channels": 3,
    "momentum": 0.9,
    "clip_update_threshold": 2.0,
    "reward_name": "softmaxlossreward",
    "reward_kwargs": {},
    "loss_name": "ce-loss",
    "activation": "relu",
    "transfer_epochs": 100,
    "batch_log": False,
    "wandb_key": "<wandb-key>",
    "disable_wandb": False,
    "transfer_training": True,
    "verbose": False,
}

TRANSFER_DATASET_NAMES = ["food11", "isic", "cub"]
MODEL_NAMES = ["vgg16", "resnet18", "vgg16bn", "resnet34"]
TRANSFER_LRS = [
    0.005,
    0.001,
]
PROPAGATOR_NAMES = ["lfp-epsilon", "vanilla-gradient"]
CLIP_UPDATES = [False]
NORM_BACKWARDS = [False]
WEIGHT_DECAYS = [0.0001]
SCHEDULER_NAMES = ["onecyclelr"]
SEEDS = [7240, 5110, 5628]

counter = 0
for transfer_dataset_name in TRANSFER_DATASET_NAMES:
    for model_name in MODEL_NAMES:
        for transfer_lr in TRANSFER_LRS:
            for propagator_name in PROPAGATOR_NAMES:
                for clip_updates in CLIP_UPDATES:
                    for weight_decay in WEIGHT_DECAYS:
                        for scheduler_name in SCHEDULER_NAMES:
                            for norm_backward in NORM_BACKWARDS:
                                for seed in SEEDS:
                                    base_config["transfer_lr"] = transfer_lr
                                    base_config["propagator_name"] = propagator_name
                                    base_config["seed"] = seed

                                    base_config["transfer_data_path"] = f"/mnt/data/{transfer_dataset_name}"
                                    base_config["transfer_dataset_name"] = transfer_dataset_name
                                    base_config["model_name"] = model_name
                                    base_config["wandb_project_name"] = (
                                        f"imagenet-to-{transfer_dataset_name}-transfer-{model_name}"
                                    )

                                    base_config["clip_updates"] = clip_updates
                                    base_config["norm_backward"] = norm_backward
                                    base_config["weight_decay"] = weight_decay
                                    base_config["scheduler_name"] = scheduler_name

                                    base_config["base_model_path"] = f"/mnt/output/{seed}/ckpts/base-model-last.pt"

                                    config_name = f"{base_config['transfer_dataset_name']}_{base_config['model_name']}_{base_config['transfer_lr']}_{base_config['propagator_name']}_{base_config['norm_backward']}_{base_config['clip_updates']}_{base_config['weight_decay']}_{base_config['scheduler_name']}_{base_config['seed']}"

                                    with open(
                                        f"{config_dir}/cluster/{config_name}_transfermodel.yaml",
                                        "w",
                                    ) as outfile:
                                        yaml.dump(
                                            base_config,
                                            outfile,
                                            default_flow_style=False,
                                        )

                                    counter += 1
print(f"Created {counter} files!")
