import contextlib
import copy
import logging
import os
import random
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import joblib
import numpy as np
import torch
import torch.nn as tnn
import torchvision
import yaml
from tqdm import tqdm

from experiment_utils.data import dataloaders, datasets, transforms
from experiment_utils.evaluation import evaluate
from experiment_utils.model import models
from experiment_utils.utils.utils import register_backward_normhooks, set_random_seeds
from lfprop.propagation import propagator_lxt as propagator
from lfprop.rewards import rewards

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyFile(object):
    def write(self, x):
        pass


def gini_idx(param):
    param = param.detach().abs()
    sortedparam, indices = torch.sort(param.view(-1))

    sortedidx = torch.arange(0, sortedparam.numel(), 1).to(param.device)

    gini_numerator = 2 * (sortedparam * sortedidx).sum()
    gini_denominator = sortedparam.numel() * sortedparam.sum()
    gini_addendum = (sortedparam.numel() + 1) / sortedparam.numel()

    return gini_numerator / gini_denominator - gini_addendum


def cosine_similarity(a, b):
    numer = (a * b).sum()
    denom = (a**2).sum() ** 0.5 * (b**2).sum() ** 0.5
    cossim = numer / denom

    return cossim


@contextlib.contextmanager
def nostdout(verbose=True):
    if verbose:
        yield
    else:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout


def get_head(model):
    if isinstance(model, torchvision.models.VGG) or isinstance(model, torchvision.models.efficientnet.EfficientNet):
        head = [m for m in model.classifier.modules() if not isinstance(m, torch.nn.Sequential)][-1]
    elif isinstance(model, torchvision.models.ResNet) or isinstance(model, torchvision.models.Inception3):
        head = model.fc
    else:
        head = model.classifier[-1]
    return head


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        batch_size,
        scheduler=None,
        lfp_composite=None,
        norm_backward=False,
        schedule_lr_every_step=False,
        clip_updates=False,
        clip_update_threshold=2.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.lfp_composite = lfp_composite
        self.norm_backward = norm_backward
        self.schedule_lr_every_step = schedule_lr_every_step
        self.clip_updates = clip_updates
        self.clip_update_threshold = clip_update_threshold
        self.global_epoch = 0
        self.global_step = 0
        self.acc_log = {"train": [], "base": [], "transfer": []}
        self.sparsity_log = {}
        self.running_update_stats = {
            "running_sum": {},
            "running_sumsq": {},
            "running_mean": {},
            "running_var": {},
        }
        self.update_log = {
            "local_mean": {},
            "local_abs_mean": {},
            "local_var": {},
            "local_abs_var": {},
            "running_var_mean": {},
            "running_l2": {},
            "cos_dist_to_running_mean": {},
            "cos_dist_to_last": {},
        }
        self.last_param_updates = {}
        self.stored_heads = {}
        self.best_acc = 0
        self.sparsity_func = gini_idx

    def fill_update_log(self):
        for name, param in self.model.named_parameters():
            if "weight" in name:
                # Initialization of Lists
                for key in self.update_log.keys():
                    if name not in self.update_log[key].keys():
                        self.update_log[key][name] = []

                for key in self.running_update_stats.keys():
                    if name not in self.running_update_stats[key]:
                        self.running_update_stats[key][name] = 0

                # Append current update statistics
                if param.grad is not None:
                    # Local Stats
                    self.update_log["local_mean"][name].append(param.grad.data.detach().mean().cpu().numpy())
                    self.update_log["local_abs_mean"][name].append(param.grad.data.detach().abs().mean().cpu().numpy())
                    self.update_log["local_var"][name].append(param.grad.data.detach().var().cpu().numpy())
                    self.update_log["local_abs_var"][name].append(param.grad.data.detach().var().cpu().numpy())

                    # Running Stats
                    self.running_update_stats["running_sum"][name] += param.grad.data.detach().view(-1)
                    self.running_update_stats["running_sumsq"][name] += param.grad.data.detach().view(-1) ** 2

                    if self.global_step > 1:
                        self.update_log["cos_dist_to_running_mean"][name].append(
                            cosine_similarity(
                                self.running_update_stats["running_mean"][name],
                                param.grad.data.detach().view(-1),
                            )
                            .cpu()
                            .numpy()
                        )
                        self.update_log["cos_dist_to_last"][name].append(
                            cosine_similarity(
                                self.last_param_updates[name],
                                param.grad.data.detach().view(-1),
                            )
                            .cpu()
                            .numpy()
                        )

                    self.running_update_stats["running_mean"][name] = (
                        self.running_update_stats["running_sum"][name] / self.global_step
                    )
                    self.running_update_stats["running_var"][name] = (
                        self.running_update_stats["running_sumsq"][name] / self.global_step
                        - self.running_update_stats["running_mean"][name] ** 2
                    )

                    self.update_log["running_var_mean"][name].append(
                        self.running_update_stats["running_var"][name].mean().cpu().numpy()
                    )
                    self.update_log["running_l2"][name].append(
                        torch.sqrt((self.running_update_stats["running_mean"][name] ** 2).sum()).cpu().numpy()
                    )

                    self.last_param_updates[name] = param.grad.data.detach().view(-1)

    def grad_step(self, inputs, labels, param_update_log=False):
        # Backward norm
        if self.norm_backward:
            norm_handles = register_backward_normhooks(self.model)
        else:
            norm_handles = []

        self.model.train()
        with torch.enable_grad():
            self.optimizer.zero_grad()
            out = self.model(inputs)

            reward = self.criterion(out, labels)
            reward.backward()

            if self.clip_updates:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_update_threshold, 2.0)

            self.optimizer.step()

        self.model.eval()

        for handle in norm_handles:
            handle.remove()

        self.global_step += 1

    def lfp_step(self, inputs, labels, param_update_log=False):
        self.model.train()

        with torch.enable_grad():
            self.optimizer.zero_grad()
            with self.lfp_composite.context(self.model) as modified:
                if self.global_step == 0:
                    print(modified)

                inputs = inputs.detach().requires_grad_(True)
                outputs = modified(inputs)

                # Calculate reward
                # Do like this to avoid tensors being kept in memory
                reward = torch.from_numpy(self.criterion(outputs, labels).detach().cpu().numpy()).to(device)

                # Write LFP Values into .grad attributes
                _ = torch.autograd.grad((outputs,), (inputs,), grad_outputs=(reward,), retain_graph=False)[0]

                for name, param in self.model.named_parameters():
                    param.grad = -param.feedback

                if self.clip_updates:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_update_threshold, 2.0)

                self.optimizer.step()

        self.model.eval()

        self.global_step += 1

    def train(
        self,
        epochs,
        dataset,
        dataset_name,
        test_dataset_base,
        test_dataset_base_name,
        test_dataset_transfer,
        test_dataset_transfer_name,
        verbose=False,
        batch_log=False,
        param_sparsity_log=False,
        param_update_log=False,
        savepath=None,
        savename="ckpt",
        saveappendage="last",
        savefrequency=1,
        fromscratch=False,
    ):
        self.store_head(dataset_name)

        loader = dataloaders.get_dataloader(
            dataset_name=dataset_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if not fromscratch and savepath:
            self.load(savepath, savename, saveappendage)

        eval_stats_train = self.eval(dataset, dataset_name)
        eval_stats_base = self.eval(test_dataset_base, test_dataset_base_name)
        eval_stats_transfer = self.eval(test_dataset_transfer, test_dataset_transfer_name)

        print(
            "Train: Initial Eval: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                float(eval_stats_train["criterion"]),
                (
                    float(eval_stats_train["accuracy_p050"])
                    if "accuracy_p050" in eval_stats_train.keys()
                    else float(eval_stats_train["micro_accuracy_top1"])
                ),
            )
        )

        print(
            "Val (Base): Initial Eval: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                float(np.mean(eval_stats_base["criterion"])),
                (
                    float(eval_stats_base["accuracy_p050"])
                    if "accuracy_p050" in eval_stats_base.keys()
                    else float(eval_stats_base["micro_accuracy_top1"])
                ),
            )
        )

        print(
            "Val (Transfer): Initial Eval: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                float(np.mean(eval_stats_transfer["criterion"])),
                (
                    float(eval_stats_transfer["accuracy_p050"])
                    if "accuracy_p050" in eval_stats_transfer.keys()
                    else float(eval_stats_transfer["micro_accuracy_top1"])
                ),
            )
        )

        logdict = {"epoch": 0}
        logdict.update({"train_" + k: v for k, v in eval_stats_train.items()})
        logdict.update({"val_base_" + k: v for k, v in eval_stats_base.items()})
        logdict.update({"val_transfer_" + k: v for k, v in eval_stats_transfer.items()})
        wandb.log(logdict)

        # Store Initial State
        if savepath and epochs > 0:
            self.save(savepath, savename, "init")

        if param_sparsity_log:
            for name, param in self.model.named_parameters():
                if name not in self.sparsity_log.keys():
                    self.sparsity_log[name] = []
                self.sparsity_log[name].append(self.sparsity_func(param).detach().cpu().numpy())

        for epoch in range(epochs):
            with tqdm(total=len(loader), disable=not verbose) as pbar:
                for index, (inp, lab) in enumerate(loader):
                    inputs = inp.to(device)
                    labels = torch.tensor(lab).to(device)

                    if self.lfp_composite is None:
                        # Grad Step
                        self.grad_step(inputs, labels, param_update_log)
                    else:
                        # LFP Step
                        self.lfp_step(inputs, labels, param_update_log)

                    if param_update_log:
                        self.fill_update_log()

                    if self.scheduler is not None and self.schedule_lr_every_step:
                        self.scheduler.step()

                    if batch_log and epoch == 0:
                        self.store_head(dataset_name)

                        eval_stats_train = self.eval(dataset, dataset_name)
                        eval_stats_base = self.eval(test_dataset_base, test_dataset_base_name)
                        eval_stats_transfer = self.eval(test_dataset_transfer, test_dataset_transfer_name)

                        self.acc_log["train"].append(
                            float(eval_stats_train["accuracy_p050"])
                            if "accuracy_p050" in eval_stats_train.keys()
                            else float(eval_stats_train["micro_accuracy_top1"])
                        )
                        self.acc_log["base"].append(
                            float(eval_stats_base["accuracy_p050"])
                            if "accuracy_p050" in eval_stats_base.keys()
                            else float(eval_stats_base["micro_accuracy_top1"])
                        )
                        self.acc_log["transfer"].append(
                            float(eval_stats_transfer["accuracy_p050"])
                            if "accuracy_p050" in eval_stats_transfer.keys()
                            else float(eval_stats_transfer["micro_accuracy_top1"])
                        )

                        wandb.log(
                            {
                                "step": index + 1,
                                "acc_log_train": (
                                    float(eval_stats_train["accuracy_p050"])
                                    if "accuracy_p050" in eval_stats_train.keys()
                                    else float(eval_stats_train["micro_accuracy_top1"])
                                ),
                                "acc_log_base": (
                                    float(eval_stats_base["accuracy_p050"])
                                    if "accuracy_p050" in eval_stats_base.keys()
                                    else float(eval_stats_base["micro_accuracy_top1"])
                                ),
                                "acc_log_transfer": (
                                    float(eval_stats_transfer["accuracy_p050"])
                                    if "accuracy_p050" in eval_stats_transfer.keys()
                                    else float(eval_stats_transfer["micro_accuracy_top1"])
                                ),
                            }
                        )

                    if param_sparsity_log:
                        for name, param in self.model.named_parameters():
                            if name not in self.sparsity_log.keys():
                                self.sparsity_log[name] = []
                            self.sparsity_log[name].append(self.sparsity_func(param).detach().cpu().numpy())

                    pbar.update(1)

            if self.scheduler is not None and not self.schedule_lr_every_step:
                self.scheduler.step()

            self.store_head(dataset_name)

            eval_stats_train = self.eval(dataset, dataset_name)
            eval_stats_base = self.eval(test_dataset_base, test_dataset_base_name)
            # print("TODO: recompute train/base stats")
            eval_stats_transfer = self.eval(test_dataset_transfer, test_dataset_transfer_name)

            print(
                "Train: Epoch {}/{}: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                    epoch + 1,
                    epochs,
                    float(eval_stats_train["criterion"]),
                    (
                        float(eval_stats_train["accuracy_p050"])
                        if "accuracy_p050" in eval_stats_train.keys()
                        else float(eval_stats_train["micro_accuracy_top1"])
                    ),
                )
            )

            print(
                "Val (Base): Epoch {}/{}: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                    epoch + 1,
                    epochs,
                    float(eval_stats_base["criterion"]),
                    (
                        float(eval_stats_base["accuracy_p050"])
                        if "accuracy_p050" in eval_stats_base.keys()
                        else float(eval_stats_base["micro_accuracy_top1"])
                    ),
                )
            )

            print(
                "Val (Transfer): Epoch {}/{}: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                    epoch + 1,
                    epochs,
                    float(eval_stats_transfer["criterion"]),
                    (
                        float(eval_stats_transfer["accuracy_p050"])
                        if "accuracy_p050" in eval_stats_transfer.keys()
                        else float(eval_stats_transfer["micro_accuracy_top1"])
                    ),
                )
            )

            logdict = {"epoch": epoch + 1}
            logdict.update({"train_" + k: v for k, v in eval_stats_train.items()})
            logdict.update({"val_base_" + k: v for k, v in eval_stats_base.items()})
            logdict.update({"val_transfer_" + k: v for k, v in eval_stats_transfer.items()})
            wandb.log(logdict)

            self.global_epoch += 1

            if savepath:
                if epoch % savefrequency == 0:
                    self.save(savepath, savename, f"ep-{epoch + 1}")
                self.save(savepath, savename, "last")

                accuracy = (
                    float(eval_stats_transfer["accuracy_p050"])
                    if "accuracy_p050" in eval_stats_transfer.keys()
                    else float(eval_stats_transfer["micro_accuracy_top1"])
                )
                if accuracy > self.best_acc:
                    self.save(savepath, savename, "best")
                    self.best_acc = accuracy

    def eval(self, dataset, dataset_name):
        self.load_evalmodel(dataset_name)

        loader = dataloaders.get_dataloader(
            dataset_name=dataset_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        print(f"Evaluating dataset '{dataset_name}' containing {len(loader)} batches")

        return_dict = evaluate.evaluate(self.eval_model, loader, self.criterion, device)

        return return_dict

    def store_head(self, dataset_name):
        head = copy.deepcopy(get_head(self.model))
        head_state_dict = copy.deepcopy(head.state_dict())

        self.stored_heads[dataset_name] = (
            head.in_features,
            head.out_features,
            head_state_dict,
        )

    def load_head(self, dataset_name):
        in_features, out_features, head_state_dict = self.stored_heads[dataset_name]
        models.replace_torchvision_last_layer(self.model, out_features)
        head = get_head(self.model)
        head.load_state_dict(head_state_dict)

        self.model.to(self.device)

    def load_evalmodel(self, dataset_name):
        self.eval_model = copy.deepcopy(self.model)
        in_features, out_features, head_state_dict = self.stored_heads[dataset_name]
        models.replace_torchvision_last_layer(self.eval_model, out_features)
        head = get_head(self.eval_model)
        head.load_state_dict(head_state_dict)

        self.eval_model.to(self.device)

    def save(self, savepath, savename, saveappendage):
        checkpoint = {
            "epoch": self.global_epoch,
            "step": self.global_step,
            "random_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "best_acc": self.best_acc,
        }
        if self.model:
            checkpoint["model"] = self.model.state_dict()
        if self.optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        if self.acc_log:
            checkpoint["acc_log"] = self.acc_log
        if self.sparsity_log:
            checkpoint["sparsity_log"] = self.sparsity_log
        if self.update_log:
            checkpoint["update_log"] = self.update_log
        if self.stored_heads:
            checkpoint["stored_heads"] = self.stored_heads

        torch.save(checkpoint, os.path.join(savepath, f"{savename}-{saveappendage}.pt"))

    def load(self, savepath, savename, saveappendage):
        if os.path.exists(os.path.join(savepath, f"{savename}-{saveappendage}.pt")):
            checkpoint = torch.load(os.path.join(savepath, f"{savename}-{saveappendage}.pt"))
            if self.model:
                self.model.load_state_dict(checkpoint["model"])
            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if "acc_log" in checkpoint:
                self.acc_log = checkpoint["acc_log"]
            if "sparsity_log" in checkpoint:
                self.sparsity_log = checkpoint["sparsity_log"]
            if "update_log" in checkpoint:
                self.update_log = checkpoint["update_log"]
            if "stored_heads" in checkpoint:
                self.stored_heads = checkpoint["stored_heads"]
            if "best_acc" in checkpoint:
                self.best_acc = checkpoint["best_acc"]
            self.global_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["step"]

            torch.set_rng_state(checkpoint["random_state"]["torch"])
            torch.cuda.set_rng_state(checkpoint["random_state"]["cuda"], device)
            np.random.set_state(checkpoint["random_state"]["numpy"])
            random.setstate(checkpoint["random_state"]["random"])

        else:
            print("No checkpoint found... not loading anything.")


def run_training_transfer(
    savepath,
    base_data_path,
    base_dataset_name,
    base_model_path,
    transfer_data_path,
    transfer_dataset_name,
    transfer_lr,
    propagator_name,
    batch_size=128,
    n_channels=3,
    momentum=0.9,
    weight_decay=0.0,
    scheduler_name="none",
    clip_updates=False,
    clip_update_threshold=2.0,
    reward_name="correct-class",
    reward_kwargs={},
    loss_name="ce-loss",
    norm_backward=True,
    transfer_epochs=5,
    model_name="cifar-vgglike",
    activation="relu",
    seed=None,
    batch_log=False,
    param_sparsity_log=False,
    param_update_log=False,
    wandb_key=None,
    disable_wandb=True,
    wandb_project_name="defaultproject",
    verbose=True,
):
    os.environ["WANDB_API_KEY"] = wandb_key
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is None:
        str_seed = "0"
    else:
        str_seed = str(seed)
    savepath = os.path.join(savepath, str_seed)
    os.makedirs(savepath, exist_ok=True)
    print("RUN:", savepath, seed)

    print("Building Paths...")

    # Wandb Path
    wandbpath = os.path.join(savepath, "wandb")
    os.makedirs(wandbpath, exist_ok=True)

    # Checkpoint Path
    ckpt_path = os.path.join(savepath, "ckpts")
    os.makedirs(ckpt_path, exist_ok=True)

    # Performance Metrics Path
    performancepath = os.path.join(savepath, "performance-metrics")
    os.makedirs(performancepath, exist_ok=True)

    # Wandb Stuff
    logdict = {
        "base_data_path": base_data_path,
        "base_dataset_name": base_dataset_name,
        "base_model_path": base_model_path,
        "transfer_data_path": transfer_data_path,
        "transfer_dataset_name": transfer_dataset_name,
        "transfer_lr": transfer_lr,
        "propagator_name": propagator_name,
        "batch_size": batch_size,
        "n_channels": n_channels,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "scheduler_name": scheduler_name,
        "clip_updates": clip_updates,
        "clip_update_threshold": clip_update_threshold,
        "reward_name": reward_name,
        "loss_name": loss_name,
        "norm_backward": norm_backward,
        "transfer_epochs": transfer_epochs,
        "model_name": model_name,
        "activation": activation,
        "seed": seed,
        "batch_log": batch_log,
    }
    logdict.update({f"reward_{k}": v for k, v in reward_kwargs.items()})

    print("Intializing wandb")
    w_id = wandb.util.generate_id()
    wandb.init(
        id=w_id,
        project=wandb_project_name,
        dir=wandbpath,
        mode="disabled" if disable_wandb else "online",
        config=logdict,
    )
    joblib.dump(w_id, os.path.join(savepath, "wandb_id.joblib"))

    # Set seeds for reproducability
    if seed is not None:
        logging.info("Setting seeds...")
        set_random_seeds(seed)

    # Data
    print("Loading Initial State...")
    with nostdout(verbose=verbose):
        test_dataset_base = datasets.get_dataset(
            base_dataset_name,
            base_data_path,
            transforms.get_transforms(base_dataset_name, "test"),
            mode="test",
        )
        train_dataset_transfer = datasets.get_dataset(
            transfer_dataset_name,
            transfer_data_path,
            transforms.get_transforms(transfer_dataset_name, "train"),
            mode="train",
        )
        test_dataset_transfer = datasets.get_dataset(
            transfer_dataset_name,
            transfer_data_path,
            transforms.get_transforms(transfer_dataset_name, "test"),
            mode="test",
        )

    # Propagation Composite
    propagation_composites = {
        "lfp-epsilon": propagator.LFPEpsilonComposite(
            norm_backward=norm_backward,
        ),
        "vanilla-gradient": None,
    }
    propagation_composite = propagation_composites[propagator_name]

    # Model
    model = models.get_model(
        model_name,
        n_channels,
        len(test_dataset_base.classes),
        device,
        replace_last_layer=True if base_dataset_name != "imagenet" else False,
        activation=activation,
    )

    # Load Base Model
    if base_dataset_name != "imagenet":
        if os.path.exists(base_model_path):
            print(base_model_path)
            checkpoint = torch.load(base_model_path)
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError(f"No checkpoint found at {base_model_path}")

    # Store base head
    head = copy.deepcopy(get_head(model))
    head_state_dict = copy.deepcopy(head.state_dict())
    base_last_layer = (head.in_features, head.out_features, head_state_dict)

    # Replace with transfer head
    if model_name in models.MODEL_MAP:
        model.classifier[-1] = tnn.Linear(model.classifier[-1].in_features, len(test_dataset_transfer.classes))
    elif model_name in models.TORCHMODEL_MAP:
        models.replace_torchvision_last_layer(model, len(test_dataset_transfer.classes))
    model.to(device)

    parameters = model.parameters()

    # Optimization
    optimizer = torch.optim.SGD(parameters, lr=transfer_lr, momentum=momentum, weight_decay=weight_decay)

    # LR Scheduling
    schedulers = {
        "none": (None, True),
        "onecyclelr": (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                transfer_lr,
                (
                    1
                    if transfer_epochs == 0
                    else transfer_epochs * int(np.ceil(len(train_dataset_transfer) / batch_size))
                ),
            ),
            True,
        ),
        "cycliclr": (
            torch.optim.lr_scheduler.CyclicLR(optimizer, transfer_lr * 0.001, transfer_lr),
            True,
        ),
        "steplr": (
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94),
            False,
        ),
    }
    scheduler, schedule_lr_every_step = schedulers[scheduler_name]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=(
            rewards.get_reward(reward_name, device, **reward_kwargs)
            if propagation_composite is not None
            else rewards.get_reward(loss_name, device, **reward_kwargs)
        ),
        device=device,
        batch_size=batch_size,
        lfp_composite=propagation_composite,
        norm_backward=norm_backward,
        schedule_lr_every_step=schedule_lr_every_step,
        clip_updates=clip_updates,
        clip_update_threshold=clip_update_threshold,
    )

    # Store base head
    trainer.stored_heads[base_dataset_name] = base_last_layer

    # Save Base Model if imagenet
    if base_dataset_name == "imagenet":
        trainer.save(savepath, "base-model", "last")

    print("Training Transfer...")
    saveappendage = "last"
    savename = "transfer-model"
    trainer.train(
        epochs=transfer_epochs,
        dataset=train_dataset_transfer,
        dataset_name=transfer_dataset_name,
        test_dataset_base=test_dataset_base,
        test_dataset_base_name=base_dataset_name,
        test_dataset_transfer=test_dataset_transfer,
        test_dataset_transfer_name=transfer_dataset_name,
        verbose=verbose,
        batch_log=batch_log,
        param_sparsity_log=param_sparsity_log,
        param_update_log=param_update_log,
        savepath=ckpt_path,
        savename=savename,
        saveappendage=saveappendage,
        savefrequency=5 if base_dataset_name == "imagenet" else 1,
        fromscratch=True,
    )

    # Eval base accuracy
    res_base = trainer.eval(test_dataset_base, base_dataset_name)

    # Eval transfer accuracy
    res_transfer = trainer.eval(test_dataset_transfer, transfer_dataset_name)

    print(
        "Accuracies Transfer: (Test1) {:.2f}, (Test2) {:.2f}".format(
            (
                float(res_base["accuracy_p050"])
                if "accuracy_p050" in res_base.keys()
                else float(res_base["micro_accuracy_top1"])
            ),
            (
                float(res_transfer["accuracy_p050"])
                if "accuracy_p050" in res_transfer.keys()
                else float(res_transfer["micro_accuracy_top1"])
            ),
        )
    )

    return trainer


def run_training_base(
    savepath,
    base_data_path,
    base_dataset_name,
    base_lr,
    propagator_name,
    batch_size=128,
    pretrained_model=True,
    n_channels=3,
    momentum=0.9,
    weight_decay=0.0,
    scheduler_name="none",
    clip_updates=False,
    clip_update_threshold=2.0,
    reward_name="correct-class",
    reward_kwargs={},
    loss_name="ce-loss",
    norm_backward=True,
    base_epochs=5,
    model_name="cifar-vgglike",
    activation="relu",
    batch_log=False,
    param_sparsity_log=False,
    param_update_log=False,
    seed=None,
    wandb_key=None,
    disable_wandb=True,
    wandb_project_name="defaultproject",
    verbose=True,
):
    os.environ["WANDB_API_KEY"] = wandb_key
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is None:
        str_seed = "0"
    else:
        str_seed = str(seed)
    savepath = os.path.join(savepath, str_seed)
    os.makedirs(savepath, exist_ok=True)
    print("RUN:", savepath, seed)

    print("Building Paths...")

    # Wandb Path
    wandbpath = os.path.join(savepath, "wandb")
    os.makedirs(wandbpath, exist_ok=True)

    # Checkpoint Path
    ckpt_path = os.path.join(savepath, "ckpts")
    os.makedirs(ckpt_path, exist_ok=True)

    # Performance Metrics Path
    performancepath = os.path.join(savepath, "performance-metrics")
    os.makedirs(performancepath, exist_ok=True)

    # Wandb Stuff
    logdict = {
        "base_data_path": base_data_path,
        "base_dataset_name": base_dataset_name,
        "base_lr": base_lr,
        "propagator_name": propagator_name,
        "batch_size": batch_size,
        "n_channels": n_channels,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "scheduler_name": scheduler_name,
        "clip_updates": clip_updates,
        "clip_update_threshold": clip_update_threshold,
        "reward_name": reward_name,
        "loss_name": loss_name,
        "norm_backward": norm_backward,
        "base_epochs": base_epochs,
        "model_name": model_name,
        "activation": activation,
        "batch_log": batch_log,
        "seed": seed,
    }
    logdict.update({f"reward_{k}": v for k, v in reward_kwargs.items()})
    print("Intializing wandb")
    w_id = wandb.util.generate_id()
    wandb.init(
        id=w_id,
        project=wandb_project_name,
        dir=wandbpath,
        mode="disabled" if disable_wandb else "online",
        config=logdict,
    )
    joblib.dump(w_id, os.path.join(savepath, "wandb_id.joblib"))

    # Set seeds for reproducability
    if seed is not None:
        logging.info("Setting seeds...")
        set_random_seeds(seed)

    # Data
    print("Loading Initial State...")
    with nostdout(verbose=verbose):
        train_dataset_base = datasets.get_dataset(
            base_dataset_name,
            base_data_path,
            transforms.get_transforms(base_dataset_name, "train"),
            mode="train",
        )
        test_dataset_base = datasets.get_dataset(
            base_dataset_name,
            base_data_path,
            transforms.get_transforms(base_dataset_name, "test"),
            mode="test",
        )

    # Propagation Composite
    propagation_composites = {
        "lfp-epsilon": propagator.LFPEpsilonComposite(
            norm_backward=norm_backward,
        ),
        "vanilla-gradient": None,
        # "lfp-zplus-zminus": propagator.LFPZplusZminusConComposite(
        #     norm_backward=norm_backward,
        #     use_input_magnitude=True,
        #     use_param_sign=False
        # ),
    }
    propagation_composite = propagation_composites[propagator_name]

    # Model
    model = models.get_model(
        model_name,
        n_channels,
        len(test_dataset_base.classes),
        device,
        replace_last_layer=True,
        activation=activation,
        pretrained_model=pretrained_model,
    )

    # Store initialization head
    head = copy.deepcopy(get_head(model))
    head_state_dict = copy.deepcopy(head.state_dict())
    base_last_layer = (head.in_features, head.out_features, head_state_dict)

    # Replace with base head
    if model_name in models.MODEL_MAP:
        model.classifier[-1] = tnn.Linear(model.classifier[-1].in_features, len(test_dataset_base.classes))
    elif model_name in models.TORCHMODEL_MAP:
        models.replace_torchvision_last_layer(model, len(test_dataset_base.classes))
    model.to(device)

    parameters = model.parameters()

    # Optimization
    optimizer = torch.optim.SGD(parameters, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    # LR Scheduling
    schedulers = {
        "none": (None, True),
        "onecyclelr": (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                base_lr,
                (1 if base_epochs == 0 else base_epochs * int(np.ceil(len(train_dataset_base) / batch_size))),
            ),
            True,
        ),
        "cycliclr": (
            torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr * 0.001, base_lr),
            True,
        ),
        "steplr": (
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94),
            False,
        ),
    }
    scheduler, schedule_lr_every_step = schedulers[scheduler_name]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=(
            rewards.get_reward(reward_name, device, **reward_kwargs)
            if propagation_composite is not None
            else rewards.get_reward(loss_name, device, **reward_kwargs)
        ),
        device=device,
        batch_size=batch_size,
        lfp_composite=propagation_composite,
        norm_backward=norm_backward,
        schedule_lr_every_step=schedule_lr_every_step,
        clip_updates=clip_updates,
        clip_update_threshold=clip_update_threshold,
    )

    # Store init head
    trainer.stored_heads[base_dataset_name] = base_last_layer

    print("Training Base Model...")
    saveappendage = "last"
    savename = "base-model"
    trainer.train(
        epochs=base_epochs,
        dataset=train_dataset_base,
        dataset_name=base_dataset_name,
        test_dataset_base=test_dataset_base,
        test_dataset_base_name=base_dataset_name,
        test_dataset_transfer=test_dataset_base,
        test_dataset_transfer_name=base_dataset_name,
        verbose=verbose,
        batch_log=batch_log,
        param_sparsity_log=param_sparsity_log,
        param_update_log=param_update_log,
        savepath=ckpt_path,
        savename=savename,
        saveappendage=saveappendage,
        fromscratch=True,
    )

    # Eval base accuracy
    res_base = trainer.eval(test_dataset_base, base_dataset_name)

    print(
        "Accuracies Base: (Test1) {:.2f}".format(
            float(res_base["accuracy_p050"])
            if "accuracy_p050" in res_base.keys()
            else float(res_base["micro_accuracy_top1"])
        )
    )

    return trainer


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_file", default="None")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print("Starting script...")

    args = get_args()
    print(f"CONFIG: {args.config_file}")
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config["config_file"] = args.config_file

    config = SimpleNamespace(**config)
    print(config)

    if config.transfer_training:
        run_training_transfer(
            savepath=config.savepath,
            base_data_path=config.base_data_path,
            base_dataset_name=config.base_dataset_name,
            base_model_path=config.base_model_path,
            transfer_data_path=config.transfer_data_path,
            transfer_dataset_name=config.transfer_dataset_name,
            transfer_lr=config.transfer_lr,
            propagator_name=config.propagator_name,
            batch_size=config.batch_size,
            n_channels=config.n_channels,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            scheduler_name=config.scheduler_name,
            clip_updates=config.clip_updates,
            clip_update_threshold=config.clip_update_threshold,
            reward_name=config.reward_name,
            reward_kwargs=config.reward_kwargs,
            loss_name=config.loss_name,
            norm_backward=config.norm_backward,
            transfer_epochs=config.transfer_epochs,
            model_name=config.model_name,
            activation=config.activation,
            batch_log=config.batch_log,
            param_sparsity_log=config.param_sparsity_log,
            param_update_log=config.param_update_log,
            seed=config.seed,
            wandb_key=config.wandb_key,
            disable_wandb=config.disable_wandb,
            wandb_project_name=config.wandb_project_name,
            verbose=config.verbose,
        )
    else:
        run_training_base(
            savepath=config.savepath,
            base_data_path=config.base_data_path,
            base_dataset_name=config.base_dataset_name,
            base_lr=config.base_lr,
            propagator_name=config.propagator_name,
            batch_size=config.batch_size,
            n_channels=config.n_channels,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            scheduler_name=config.scheduler_name,
            clip_updates=config.clip_updates,
            clip_update_threshold=config.clip_update_threshold,
            reward_name=config.reward_name,
            reward_kwargs=config.reward_kwargs,
            loss_name=config.loss_name,
            norm_backward=config.norm_backward,
            base_epochs=config.base_epochs,
            model_name=config.model_name,
            activation=config.activation,
            batch_log=config.batch_log,
            param_sparsity_log=config.param_sparsity_log,
            param_update_log=config.param_update_log,
            seed=config.seed,
            wandb_key=config.wandb_key,
            disable_wandb=config.disable_wandb,
            wandb_project_name=config.wandb_project_name,
            verbose=config.verbose,
        )
