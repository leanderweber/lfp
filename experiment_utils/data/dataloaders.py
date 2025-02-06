import torch.utils.data as tdata

DATALOADER_MAPPING = {
    "food11": tdata.DataLoader,
    "imagenet": tdata.DataLoader,
    "cub": tdata.DataLoader,
    "isic": tdata.DataLoader,
    "mnist": tdata.DataLoader,
    "cifar10": tdata.DataLoader,
    "cifar100": tdata.DataLoader,
    "circles": tdata.DataLoader,
    "blobs": tdata.DataLoader,
    "swirls": tdata.DataLoader,
}


def get_dataloader(dataset_name, dataset, batch_size, shuffle):
    """
    selects the correct dataloader for the dataset
    """

    # Check if dataset_name is valid
    if dataset_name not in DATALOADER_MAPPING:
        raise ValueError("Dataloader for dataset '{}' not supported.".format(dataset_name))

    # Load correct dataloader
    dataloader = DATALOADER_MAPPING[dataset_name](
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Return dataset
    return dataloader
