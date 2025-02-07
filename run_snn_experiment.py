import argparse
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiment_utils.data import datasets, transforms
from experiment_utils.training import get_training_method


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}
    return config


def get_args():
    parser = argparse.ArgumentParser(description="Compute relevances and activations")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/spiking_neural_networks/cluster/one_cycle_lfp_snn_lifcnn_10879_0.2_25_beta_0.9_correct-class-spikes-ratecoded_True_True_sgd_0.0.yaml",
    )
    return parser.parse_args()


def main():
    print("*** load config ***")
    args = get_args()
    config_path = args.config_file
    print("Using Config at {}".format(config_path))
    config = load_config(config_path)
    config["wandb_tag"] = config["wandb_tag"] + "_" + ".".join(config_path.split("/")[-1].split(".")[:-1])
    print("*** start run ***")
    start_model_training(config, config_name=".".join(config_path.split("/")[-1].split(".")[:-1]))


def start_model_training(config, config_name):
    random_seed = config.get("random_seed", 666)
    num_epochs = config.get("num_epochs", 10)

    dataset_name = config.get("dataset_name")
    data_path = config.get("data_path")
    batch_size = config.get("batch_size")
    optim_name = config.get("optimizer_name", "sgd")
    lr = config.get("lr", 0.01)
    weight_decay = config.get("weight_decay", 0.0)
    train_method = config.get("train_method", "multi_stage_lfp_snn")

    pl.seed_everything(random_seed)

    # --- init data
    train_ds = datasets.get_dataset(
        dataset_name=dataset_name,
        root_path=data_path,
        transform=transforms.get_transforms(dataset_name=dataset_name, mode="train"),
        mode="train",
    )
    test_ds = datasets.get_dataset(
        dataset_name=dataset_name,
        root_path=data_path,
        transform=transforms.get_transforms(dataset_name=dataset_name, mode="test"),
        mode="test",
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    # --- init pl model

    pl_model = get_training_method(train_method)(config, config_name)
    pl_model.set_optimizer(optim_name, pl_model.parameters(), lr, weight_decay)

    # --- init trainer

    if config["wandb_name"]:
        wandb_api_key = config.get("wandb_api_key")
        wandb_project_name = config["wandb_name"]
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb_id = f"{config_name}" if config.get("unique_wandb_ids", False) else None
        logger_ = WandbLogger(
            project=wandb_project_name,
            name=f"{config_name}",
            id=wandb_id,
            config=config,
        )

    trainer = Trainer(
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
        devices=1,
        max_epochs=num_epochs,
        accelerator="gpu",
        logger=logger_,
    )
    # --- train
    trainer.validate(pl_model, test_dl)
    trainer.fit(pl_model, train_dl, test_dl)


if __name__ == "__main__":
    main()
