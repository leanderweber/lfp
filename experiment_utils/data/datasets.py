import os

import torchvision.datasets as tvisiondata

from . import custom_datasets

DATASET_MAPPING = {
    "food11": custom_datasets.Food11,
    "imagenet": custom_datasets.ImageNet,
    "cub": custom_datasets.CUB,
    "isic": custom_datasets.ISIC,
    "mnist": tvisiondata.MNIST,
    "cifar10": tvisiondata.CIFAR10,
    "cifar100": tvisiondata.CIFAR100,
    "circles": custom_datasets.SKLearnCircles,
    "blobs": custom_datasets.SKLearnBlobs,
    "swirls": custom_datasets.Swirls,
}


def get_dataset(dataset_name, root_path, transform, mode, **kwargs):
    """
    gets the specified dataset and saves it
    """

    # Check if mode is valid
    if mode not in ["train", "test"]:
        raise ValueError("Mode '{}' not supported. Mode needs to be one of 'train', 'test'".format(mode))

    # Map mode (kinda illegal but so that imagenet works)
    if (dataset_name == "imagenet") and mode == "test":
        mode = "val"

    # Check if dataset_name is valid
    if dataset_name not in DATASET_MAPPING:
        raise ValueError("Dataset '{}' not supported.".format(dataset_name))

    # Adapt root_path
    if DATASET_MAPPING[dataset_name] not in [
        custom_datasets.ImageNet,
        custom_datasets.CUB,
        custom_datasets.ISIC,
        custom_datasets.SKLearnCircles,
        custom_datasets.SKLearnBlobs,
        custom_datasets.Swirls,
        custom_datasets.Food11,
    ]:
        root = os.path.join(root_path, dataset_name)
    else:
        root = root_path

    # Load correct dataset
    if dataset_name not in ["mnist", "cifar10", "cifar100"]:
        dataset = DATASET_MAPPING[dataset_name](
            root=root,
            transform=transform,
            **{
                **kwargs,
                **{
                    "download": True,
                    "train": mode == "train",
                    "mode": mode,
                },
            },
        )
    else:
        dataset = DATASET_MAPPING[dataset_name](
            root=root,
            transform=transform,
            download=True,
            train=mode == "train",
        )

    # Return dataset
    return dataset
