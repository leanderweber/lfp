import logging
from typing import Optional, TypeVar

import torch

TRecall = TypeVar("TRecall")
import torcheval
import torcheval.metrics
import torcheval.metrics.classification


# Recall patch: The multiclass recall of torcheval is currently bugged, see https://github.com/pytorch/torcheval/issues/150
# This patches the bug
def _recall_compute(
    num_tp: torch.Tensor,
    num_labels: torch.Tensor,
    num_predictions: torch.Tensor,
    average: Optional[str],
) -> torch.Tensor:
    if average in ("macro", "weighted"):
        # Ignore classes which have no samples in `target` and `input`
        mask = (num_labels != 0) | (num_predictions != 0)
        num_tp = num_tp[mask]
        num_labels = num_labels[mask]  # THIS IS THE PATCH

    recall = num_tp / num_labels

    isnan_class = torch.isnan(recall)
    if isnan_class.any():
        nan_classes = isnan_class.nonzero(as_tuple=True)[0]
        logging.warning(
            f"One or more NaNs identified, as no ground-truth instances of "
            f"{nan_classes.tolist()} have been seen. These have been converted to zero."
        )
        recall = torch.nan_to_num(recall)

    if average == "micro":
        return recall
    elif average == "macro":
        return recall.mean()
    elif average == "weighted":
        # pyre-fixme[61]: `mask` is undefined, or not always defined.
        weights = num_labels[mask] / num_labels.sum()
        return (recall * weights).sum()
    else:  # average is None
        return recall


@torch.inference_mode()
def compute(self: TRecall) -> torch.Tensor:
    """
    Return the recall score.

    NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.
    """
    return _recall_compute(self.num_tp, self.num_labels, self.num_predictions, self.average)


torcheval.metrics.classification.MulticlassRecall.compute = compute


def eval(model, loader, criterion_func, device):
    """
    Evaluates one epoch (predictions and accuracy). Returns labels, predictions, accuracy and reward function.
    """
    binary = True
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        if outputs.shape[-1] == 1:
            binary = True
            num_classes = 2
        else:
            binary = False
            num_classes = outputs.shape[-1]

    if binary:
        metrics = {
            "criterion": torcheval.metrics.Mean(device=device),
            "accuracy_p040": torcheval.metrics.BinaryAccuracy(threshold=0.4, device=device),
            "accuracy_p050": torcheval.metrics.BinaryAccuracy(threshold=0.5, device=device),
            "accuracy_p060": torcheval.metrics.BinaryAccuracy(threshold=0.6, device=device),
            "precision_p040": torcheval.metrics.BinaryPrecision(threshold=0.4, device=device),
            "precision_p050": torcheval.metrics.BinaryPrecision(threshold=0.5, device=device),
            "precision_p060": torcheval.metrics.BinaryPrecision(threshold=0.6, device=device),
            "recall_p040": torcheval.metrics.BinaryRecall(threshold=0.4, device=device),
            "recall_p050": torcheval.metrics.BinaryRecall(threshold=0.5, device=device),
            "recall_p060": torcheval.metrics.BinaryRecall(threshold=0.6, device=device),
            "f1_p040": torcheval.metrics.BinaryF1Score(threshold=0.4, device=device),
            "f1_p050": torcheval.metrics.BinaryF1Score(threshold=0.5, device=device),
            "f1_p060": torcheval.metrics.BinaryF1Score(threshold=0.6, device=device),
        }
    else:
        metrics = {
            "criterion": torcheval.metrics.Mean(device=device),
            "micro_accuracy_top1": torcheval.metrics.MulticlassAccuracy(
                average="micro", num_classes=num_classes, k=1, device=device
            ),
            "micro_accuracy_top3": torcheval.metrics.MulticlassAccuracy(
                average="micro", num_classes=num_classes, k=3, device=device
            ),
            "micro_accuracy_top5": torcheval.metrics.MulticlassAccuracy(
                average="micro", num_classes=num_classes, k=5, device=device
            ),
            "micro_precision": torcheval.metrics.MulticlassPrecision(
                average="micro", num_classes=num_classes, device=device
            ),
            "micro_recall": torcheval.metrics.MulticlassRecall(average="micro", num_classes=num_classes, device=device),
            "micro_f1": torcheval.metrics.MulticlassF1Score(average="micro", num_classes=num_classes, device=device),
            "macro_accuracy_top1": torcheval.metrics.MulticlassAccuracy(
                average="macro", num_classes=num_classes, k=1, device=device
            ),
            "macro_accuracy_top3": torcheval.metrics.MulticlassAccuracy(
                average="macro", num_classes=num_classes, k=3, device=device
            ),
            "macro_accuracy_top5": torcheval.metrics.MulticlassAccuracy(
                average="macro", num_classes=num_classes, k=5, device=device
            ),
            "macro_precision": torcheval.metrics.MulticlassPrecision(
                average="macro", num_classes=num_classes, device=device
            ),
            "macro_recall": torcheval.metrics.MulticlassRecall(average="macro", num_classes=num_classes, device=device),
            "macro_f1": torcheval.metrics.MulticlassF1Score(average="macro", num_classes=num_classes, device=device),
        }

    # Set model to eval mode
    model.eval()

    # Iterate over data.
    for i, (inputs, labels) in enumerate(loader):
        # Prepare inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            # Get model predictions
            outputs = model(inputs)

        with torch.set_grad_enabled(True):
            # Get rewards
            if isinstance(criterion_func, torch.nn.modules.loss._Loss):
                crit = torch.ones_like(outputs) * criterion_func(outputs, labels)  # reshape to correct shape
            else:
                crit = criterion_func(outputs, labels)

        if binary:
            outputs = torch.nn.functional.sigmoid(outputs).squeeze()

        for k, v in metrics.items():
            if k == "criterion":
                metrics[k].update(crit)
            else:
                metrics[k].update(outputs, labels)

    return_dict = {m: metric.compute().detach().cpu().numpy() for m, metric in metrics.items()}

    # Return labels, predictions, accuracy and loss
    return return_dict
