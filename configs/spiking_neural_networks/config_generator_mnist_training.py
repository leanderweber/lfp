import copy
import os
import shutil
from pathlib import Path

import yaml

config_dir = Path("configs/spiking_neural_networks")

# Create a backup directory named after the current date and time
# backup_dir = os.path.join(
#     "configs/backups",
#     os.path.basename(os.path.dirname(config_dir)),
#     f"{os.path.basename(config_dir)}_backup_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
# )
# shutil.copytree(config_dir, backup_dir, ignore=shutil.ignore_patterns("*.py"), dirs_exist_ok=True)

# Remove the original directory
shutil.rmtree(config_dir / "cluster", onerror=lambda a, b, c: None)
shutil.rmtree(config_dir / "local", onerror=lambda a, b, c: None)
os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)


default = {
    "batch_size": 128,
    "beta": 0.9,
    "clip_update": True,
    "data_path": "./data",
    "dataset_name": "mnist",
    "num_epochs": 5,
    "lif_threshold": 1,
    "lr": 0.2,
    "minmem": None,
    "model_name": "lifcnn",
    "momentum": 0.9,
    "n_channels": 1,
    "n_outputs": 10,
    "no_surrogate": True,
    "norm_backward": True,
    "optimizer_name": "sgd",
    "out_name": "run",
    "propagator_name": "lfp-epsilon",
    "random_seed": 8359,
    "reward_name": "correct-class-spikes-ratecoded",
    "savepath": "./checkpoints",
    "show_tqdm": True,
    "shuffle": True,
    "spike_encoding": "rate",
    "start_epoch": 0,
    "steps_per_prediction": 25,
    "wandb_name": "[lfp-snn] Experiments",
    "wandb_api_key": "",  # ADD key here
    "wandb_tag": "",  # ADD tag here
    "train_method": "one_cycle_lfp_snn",
}


def store_config(config, config_name):
    filename = os.path.join(config_dir, "cluster", f"{config_name}.yaml")
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    print(f"Stored configuration: {filename}")


config_count = 0

for steps_per_prediction in [25]:
    for lr in [
        # 1e1,  # added after sebastian's review
        # (1e1 + 1e0) / 2,  # added after sebastian's review
        # 2 * 1e0,  # added after sebastian's review
        # 1e0,  # added after sebastian's review
        # 1e-4,
        # 1e-3,
        # 1e-2,
        # investigate range between 0.1 and 1
        i * 1e-1
        for i in range(2, 3)
        # 0.2,
    ]:
        for weight_decay in [0.0, 0.001, 0.001, 0.005]:
            for batch_size in [128]:
                for norm_backward in [True]:
                    for clip_update in [True]:
                        for epochs in [10]:
                            for optimizer_name in ["sgd"]:
                                for random_seed in [10879, 8359, 26347]:
                                    for beta in [0.9]:
                                        for model_name in ["lifcnn", "lifmlp"]:
                                            for train_method in ["one_cycle_lfp_snn", "one_cycle_grad_snn"]:
                                                for reward_name in ["correct-class-spikes-ratecoded"]:
                                                    config = copy.deepcopy(default)

                                                    config["lr"] = lr
                                                    config["batch_size"] = batch_size
                                                    config["epochs"] = epochs
                                                    config["beta"] = beta
                                                    config["steps_per_prediction"] = steps_per_prediction
                                                    config["reward_name"] = reward_name
                                                    config["optimizer_name"] = optimizer_name
                                                    config["norm_backward"] = norm_backward
                                                    config["clip_update"] = clip_update
                                                    config["random_seed"] = random_seed
                                                    config["train_method"] = train_method
                                                    config["model_name"] = model_name
                                                    config["weight_decay"] = weight_decay
                                                    if "mlp" in model_name:
                                                        config["n_channels"] = 784

                                                    config_name = "_".join(
                                                        [
                                                            train_method,
                                                            model_name,
                                                            str(random_seed),
                                                            str(lr),
                                                            str(steps_per_prediction),
                                                            "beta",
                                                            str(beta),
                                                            reward_name,
                                                            str(norm_backward),
                                                            str(clip_update),
                                                            optimizer_name,
                                                            str(weight_decay),
                                                        ]
                                                    )

                                                    store_config(config, config_name)
                                                    config_count += 1

print(f"Total configurations generated: {config_count}")
