import numpy as np
import pytorch_lightning as pl
import torch
from snntorch import functional as SF
from snntorch import surrogate

import lfprop.model.spiking_networks as models
import lfprop.propagation.propagator_snn as propagator
from experiment_utils.data import dataloaders, datasets, transforms
from experiment_utils.training.helper import get_optimizer
from experiment_utils.training.metrics import MultiClassMetrics
from lfprop import rewards

# Assuming these are custom modules from the original script


class SNNModel(pl.LightningModule):
    def __init__(self, config, config_name):
        super().__init__()
        model_name = config["model_name"]
        n_channels = config["n_channels"]
        n_outputs = config["n_outputs"]
        beta = config["beta"]
        minmem = config["minmem"]
        lif_threshold = config["lif_threshold"]
        reward_name = config["reward_name"]
        steps_per_prediction = config["steps_per_prediction"]
        spike_encoding = config["spike_encoding"]
        clip_update = config["clip_update"]
        optimizer_name = config["optimizer_name"]
        lr = config["lr"]
        momentum = config["momentum"]
        norm_backward = config["norm_backward"]

        self.automatic_optimization = False  # necessary for lfp updates
        self.save_hyperparameters()

        self.model = models.get_model(
            model_name,
            n_channels,
            n_outputs,
            self.device,
            beta=beta,
            minmem=minmem,
            threshold=lif_threshold,
        )
        self.reward_func = rewards.get_reward(reward_name=reward_name, device=self.device)
        self.reward_name = reward_name

        self.spike_encoding = spike_encoding

        self.clip_update = clip_update

        self.steps_per_prediction = steps_per_prediction

        self.optimizer_name = optimizer_name
        self.optim = None
        self.lr = lr

        self.momentum = momentum
        self.compute_metrics = {
            "train": MultiClassMetrics(),
            "valid": MultiClassMetrics(),
        }

        self.lfp_propagator = propagator.LRPRewardPropagator(self.model, norm_backward)

    def setup(self, stage):
        self.reward_func.device = self.device

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.model.reset()  # NOTE releases the accumulated states
        self.lfp_propagator.reset()  # NOTE releases the accumulated feedback

        u_rec = []
        spk_rec = []

        for step in range(self.steps_per_prediction):
            spk_out, u_out = self(inputs)
            u_rec.append(u_out.detach_())
            spk_rec.append(spk_out.detach_())
        spikes = torch.stack(spk_rec, dim=0)
        tensions = torch.stack(u_rec, dim=0)

        if "spike" in self.reward_name:
            reward = self.reward_func(spikes=spikes, potentials=tensions, labels=labels)
            # ╰─ n_steps x batch_size x n_classes
            preds = spikes.sum(0).argmax(-1)  # sum over steps → argmax wrt to class
        elif "membrane" in self.reward_name:
            reward = self.reward_func(spikes=spikes, potentials=tensions, labels=labels)
            preds = None
        else:
            raise ValueError("Only `spike` and `membrane` are supported for the reward inputs.")

        with torch.no_grad():
            for step in range(self.steps_per_prediction):
                # go backwards through sequence and write reward into accumulated_feedback param attr
                self.lfp_propagator.propagate(iteration_feedback=reward[-(step + 1)], iteration_idx=step)

        for name, param in self.model.named_parameters():
            if hasattr(param, "accumulated_feedback"):
                # overwrite grad with lfp-signal
                param.grad = -param.accumulated_feedback
            else:
                print("\nWARNING: {} has no accumulated feedback!".format(name))

        # clip rewards
        models.clip_gradients(self.model, self.clip_update, clip_update_threshold=0.06)

        params_before = {n: p.clone() for n, p in self.model.named_parameters()}
        rewards = {n: p.grad.clone() for n, p in self.model.named_parameters()}

        # Perform optimization step (self.automatic_optimization=False is necessary for this!)
        optimizer = self.optimizers()
        optimizer.step()
        optimizer.zero_grad()

        params_after = {n: p.clone() for n, p in self.model.named_parameters()}
        params_diff = {n: (params_before[n] - p.data).pow(2).sum() for n, p in self.model.named_parameters()}
        params_mean_diff = {n: (params_before[n] - p.data).mean() for n, p in self.model.named_parameters()}

        log_dict = {}

        # log reward signal
        log_dict.update(
            {
                **{f"reward/{n}_mean": r.mean() for n, r in rewards.items()},
                **{f"reward/{n}_min": r.min() for n, r in rewards.items()},
                **{f"reward/{n}_max": r.max() for n, r in rewards.items()},
                **{f"reward/{n}_std": r.std() for n, r in rewards.items()},
                **{f"reward/{n}_non_zero_ratio": (r.abs() < 1e-6).sum() / r.shape.numel() for n, r in rewards.items()},
            }
        )
        # log params
        log_dict.update(
            {
                **{f"params/{n}_abs_mean": p.abs().mean() for n, p in params_after.items()},
                **{f"params/{n}_abs_min": p.abs().min() for n, p in params_after.items()},
                **{f"params/{n}_abs_max": p.abs().max() for n, p in params_after.items()},
                **{f"params/{n}_abs_std": p.abs().std() for n, p in params_after.items()},
                **{f"params/{n}_mean": p.mean() for n, p in params_after.items()},
                **{f"params/{n}_min": p.min() for n, p in params_after.items()},
                **{f"params/{n}_max": p.max() for n, p in params_after.items()},
                **{f"params/{n}_std": p.std() for n, p in params_after.items()},
                **{
                    f"params/{n}_non_zero_ratio": (p.abs() < 1e-6).sum() / p.shape.numel()
                    for n, p in params_after.items()
                },
                **{f"params/{n}_diff": d for n, d in params_diff.items()},
                **{f"params/{n}_mean_diff": d for n, d in params_mean_diff.items()},
            }
        )

        # log reward-vals
        init_corr_cls_rew = reward[:, torch.arange(len(batch[1])), batch[1]]
        log_dict.update(
            {
                "init_rew_vals/mean_corr_cls": init_corr_cls_rew.mean(),
                **{
                    f"init_rew_vals/corr_cls_step_{i}": init_corr_cls_rew[i].mean(-1)
                    for i in range(init_corr_cls_rew.shape[0])
                },
            }
        )

        spikes_corr_cls = spikes[:, torch.arange(len(batch[1])), batch[1]]
        spikes_all_cls = spikes.sum(-1)
        ratio_corr_vs_others = spikes_corr_cls / (spikes_all_cls - spikes_corr_cls + 1e-6)
        mean_ratio_corr_vs_others = ratio_corr_vs_others.mean(-1)
        mean_spikes = spikes.mean(-1).mean(-1)
        log_dict.update(
            {
                **{f"spikes/avg_num_spikes_{i}": mean_spikes[i] for i in range(len(mean_spikes))},
                "spikes/avg_num_spikes": spikes.mean(),
                "spikes/corr_cls_div_others_mean": mean_ratio_corr_vs_others.mean(),
                **{
                    f"spikes/corr_cls_div_others_{i}": mean_ratio_corr_vs_others[i]
                    for i in range(len(mean_ratio_corr_vs_others))
                },
            }
        )

        # batch accuracy
        log_dict.update({"batch_acc": ((preds == batch[1]).sum() / len(preds))})

        self.log_dict(log_dict, sync_dist=True, on_step=True, on_epoch=False, prog_bar=False)
        if "rate" in self.spike_encoding.lower():
            self.log_results(spikes.sum(0), labels, spikes, reward=reward, stage="train")

    def log_results(self, preds, labels, spikes, reward, stage):
        metrics = self.compute_metrics[stage](preds, labels)
        rel_spikes = spikes.mean()
        mean_reward = reward.mean() if reward != None else 0.0
        self.log_dict(
            {
                f"{stage}_rel_spk": rel_spikes,
                f"{stage}_mean_reward": mean_reward,
            }
            | {f"{stage}_{key}": val for key, val in metrics.items()},
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        spk_rec = []
        for step in range(self.steps_per_prediction):
            spk_out, _ = self(inputs)
            spk_rec.append(spk_out)
        spikes = torch.stack(spk_rec, dim=0)
        # logging
        if "rate" in self.spike_encoding.lower():
            preds = spikes.sum(0)
            self.log_results(preds, labels, spikes, reward=None, stage="valid")

    def configure_optimizers(self, milestones=None):
        # milestones = self.config.get("milestones", milestones)
        if milestones is None:
            milestones = [50, 75]
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=milestones, gamma=0.1)
        # sche = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=95)
        # warmup = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=0.01, total_iters=5)
        # sche = torch.optim.lr_scheduler.SequentialLR(self.optim, schedulers=[warmup, sche], milestones=[5])
        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
        }

        return [self.optim], [scheduler]

    def set_optimizer(self, optim_name, params, lr, weight_decay=0.0):
        self.lr = lr
        self.optim = get_optimizer(optim_name, params, lr, weight_decay)

    def collect_and_log_debug_reward_info(self):
        param_rew_nonzero_percent = {}
        param_rew_abs_mean = {}
        param_abs_mean = {}
        for name, param in self.model.named_parameters():
            if hasattr(param, "grad"):
                # ╭─ WandB logging ──────────────────────────────────────────────────────────────────╮
                param_rew_nonzero_percent[name] = (
                    (torch.count_nonzero(param.grad) / param.grad.numel()).detach().cpu().numpy()
                )
                param_rew_abs_mean[name] = (param.grad.abs().mean()).detach().cpu().numpy()
                param_abs_mean[name] = (param.data.abs().mean()).detach().cpu().numpy()
                # ╰──────────────────────────────────────────────────────────────────────────────────╯

                # models.clip_gradients(self.model, clip_update, clip_update_threshold=0.06)

                # ╭─ WandB logging ──────────────────────────────────────────────────────────────────╮
                param_rew_nonzero_percent["ac_" + name] = (
                    (torch.count_nonzero(param.grad) / param.grad.numel()).detach().cpu().numpy()
                )
                param_rew_abs_mean["ac_" + name] = (param.grad.abs().mean()).detach().cpu().numpy()
                param_abs_mean["ac_" + name] = (param.data.abs().mean()).detach().cpu().numpy()
                # ╰──────────────────────────────────────────────────────────────────────────────────╯
        param_rew_nonzero_percent["mean"] = np.mean(
            [param_rew_nonzero_percent[k] for k in param_rew_nonzero_percent.keys() if "ac_" not in k]
        )
        param_rew_abs_mean["mean"] = np.mean(
            [param_rew_abs_mean[k] for k in param_rew_abs_mean.keys() if "ac_" not in k]
        )
        param_abs_mean["mean"] = np.mean([param_abs_mean[k] for k in param_abs_mean.keys() if "ac_" not in k])
        self.log_dict(
            {f"debug nonzero-rew % / {key}": float(val) for key, val in param_rew_nonzero_percent.items()}
            | {f"debug abs-rew |d| / {key}": float(val) for key, val in param_rew_abs_mean.items()}
            | {f"debug abs-params |p| /{key}": float(val) for key, val in param_abs_mean.items()},
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )


class OneCycleSNNModel(SNNModel):
    def __init__(self, config, config_name):
        super().__init__(config, config_name)

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        scheduler.step()  # OneCycel LR scheduler is steped after/before each batch
        super().training_step(batch, batch_idx)

    def configure_optimizers(self):
        stepping_batches = self.trainer.estimated_stepping_batches
        sche = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.lr, total_steps=stepping_batches)

        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
            "interval": "step",  # Step the scheduler after every batch
            "frequency": 1,
        }
        return [self.optim], [scheduler]


class GradSNNModel(SNNModel):
    def __init__(self, config, config_name):
        super().__init__(config, config_name)
        model_name = config["model_name"]
        n_channels = config["n_channels"]
        n_outputs = config["n_outputs"]
        beta = config["beta"]

        lif_threshold = config["lif_threshold"]

        spike_encoding = config["spike_encoding"]

        # Replace the model with a gradient-enabled version
        spike_grad = surrogate.fast_sigmoid(slope=25)
        grad_model_kwargs = {
            "beta": beta,
            "threshold": lif_threshold,
            "spike_grad": spike_grad,
        }
        self.model = models.get_model("grad" + model_name, n_channels, n_outputs, self.device, **grad_model_kwargs)

        # Set up loss function
        if "rate" in spike_encoding.lower():
            self.loss_fn = SF.loss.ce_rate_loss()
        elif "count" in spike_encoding.lower():
            self.loss_fn = SF.loss.ce_count_loss()
        else:
            raise ValueError("Only `count` and `rate` are currently supported for the `loss_fn`.")

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.model.train()
        self.model.reset()

        optimizer = self.optimizers()
        optimizer.zero_grad()

        spk_rec = []
        u_rec = []

        for step in range(self.steps_per_prediction):
            spk_out, u_out = self(inputs)
            spk_rec.append(spk_out)
            u_rec.append(u_out)

        spikes = torch.stack(spk_rec)

        loss = self.loss_fn(spk_out=spikes, targets=labels)
        self.manual_backward(loss)

        models.clip_gradients(self.model, self.clip_update, clip_update_threshold=0.06)

        params_before = {n: p.clone() for n, p in self.model.named_parameters()}
        rewards = {n: p.grad.clone() for n, p in self.model.named_parameters()}

        optimizer.step()

        # Logging

        params_after = {n: p.clone() for n, p in self.model.named_parameters()}
        params_diff = {n: (params_before[n] - p.data).pow(2).sum() for n, p in self.model.named_parameters()}
        params_mean_diff = {n: (params_before[n] - p.data).mean() for n, p in self.model.named_parameters()}

        log_dict = {}

        # log reward signal
        log_dict.update(
            {
                **{f"reward/{n}_mean": r.mean() for n, r in rewards.items()},
                **{f"reward/{n}_min": r.min() for n, r in rewards.items()},
                **{f"reward/{n}_max": r.max() for n, r in rewards.items()},
                **{f"reward/{n}_std": r.std() for n, r in rewards.items()},
                **{f"reward/{n}_non_zero_ratio": (r.abs() < 1e-6).sum() / r.shape.numel() for n, r in rewards.items()},
            }
        )
        # log params
        log_dict.update(
            {
                **{f"params/{n}_mean": p.mean() for n, p in params_after.items()},
                **{f"params/{n}_min": p.min() for n, p in params_after.items()},
                **{f"params/{n}_max": p.max() for n, p in params_after.items()},
                **{f"params/{n}_std": p.std() for n, p in params_after.items()},
                **{
                    f"params/{n}_non_zero_ratio": (p.abs() < 1e-6).sum() / p.shape.numel()
                    for n, p in params_after.items()
                },
                **{f"params/{n}_diff": d for n, d in params_diff.items()},
                **{f"params/{n}_mean_diff": d for n, d in params_mean_diff.items()},
            }
        )

        spikes_corr_cls = spikes[:, torch.arange(len(batch[1])), batch[1]]
        spikes_all_cls = spikes.sum(-1)
        ratio_corr_vs_others = spikes_corr_cls / (spikes_all_cls - spikes_corr_cls + 1e-6)
        mean_ratio_corr_vs_others = ratio_corr_vs_others.mean(-1)
        mean_spikes = spikes.mean(-1).mean(-1)
        log_dict.update(
            {
                **{f"spikes/avg_num_spikes_{i}": mean_spikes[i] for i in range(len(mean_spikes))},
                "spikes/avg_num_spikes": spikes.mean(),
                "spikes/corr_cls_div_others_mean": mean_ratio_corr_vs_others.mean(),
                **{
                    f"spikes/corr_cls_div_others_{i}": mean_ratio_corr_vs_others[i]
                    for i in range(len(mean_ratio_corr_vs_others))
                },
            }
        )

        # batch accuracy
        log_dict.update({"batch_acc": ((spikes.sum(0).argmax(-1) == batch[1]).sum() / len(spikes.sum(0)))})

        self.log_dict(log_dict, sync_dist=True, on_step=True, on_epoch=False, prog_bar=False)
        if "rate" in self.spike_encoding.lower():
            self.log_results(spikes.sum(0), labels, spikes, reward=None, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        self.model.eval()
        self.model.reset()

        spk_rec = []

        with torch.no_grad():
            for step in range(self.steps_per_prediction):
                spk_out, _ = self(inputs)
                spk_rec.append(spk_out)

        spikes = torch.stack(spk_rec, dim=0)

        loss = self.loss_fn(spk_out=spikes, targets=labels)

        # Logging
        preds = spikes.sum(0)
        self.log_results(preds, labels, spikes, reward=None, stage="valid")
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # return loss

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"{self.optimizer_name} is not supported.")

        return optimizer


class OneCycleGradSNNModel(GradSNNModel):
    def __init__(self, config, config_name):
        super().__init__(config, config_name)

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        scheduler.step()  # OneCycel LR scheduler is steped after/before each batch
        super().training_step(batch, batch_idx)

    def configure_optimizers(self):
        stepping_batches = self.trainer.estimated_stepping_batches
        sche = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.lr, total_steps=stepping_batches)

        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
            "interval": "step",  # Step the scheduler after every batch
            "frequency": 1,
        }
        return [self.optim], [scheduler]


class SNNDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, data_path, batch_size, shuffle, num_workers=8):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset_train = datasets.get_dataset(
            dataset_name=self.dataset_name,
            root_path=self.data_path,
            transform=transforms.get_transforms(self.dataset_name, "train"),
            mode="train",
        )
        self.dataset_test = datasets.get_dataset(
            dataset_name=self.dataset_name,
            root_path=self.data_path,
            transform=transforms.get_transforms(self.dataset_name, "test"),
            mode="test",
        )

    def train_dataloader(self):
        return dataloaders.get_dataloader(
            dataset_name=self.dataset_name,
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return dataloaders.get_dataloader(
            dataset_name=self.dataset_name,
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # self.shuffle,
        )
