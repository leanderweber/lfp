from torchmetrics.classification import Accuracy, F1Score, Precision, Recall


class MultiClassMetrics:
    def __init__(self, num_labels=None):
        self.num_labels = num_labels
        if num_labels:
            self._init_metrics(self.num_labels)

    def _init_metrics(self, num_classes, device="cuda"):
        print("\n→ Metrics: {}\n".format(num_classes))
        self.metrics = {
            "acc": Accuracy(task="multiclass", num_classes=num_classes).to(device),
            "precision": Precision(task="multiclass", num_classes=num_classes).to(device),
            "recall": Recall(task="multiclass", num_classes=num_classes).to(device),
            "f1": F1Score(task="multiclass", num_classes=num_classes).to(device),
        }

    def __call__(self, predictions, labels):
        if self.num_labels is None:
            self.num_labels = predictions.shape[-1]
            self._init_metrics(predictions.shape[-1], labels.device)
        return {name: metric(predictions, labels) for name, metric in self.metrics.items()}
