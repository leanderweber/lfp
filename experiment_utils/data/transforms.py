import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if np.random.choice([0, 1]):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class DoubleCompose(T.Compose):
    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class DoubleResize(torch.nn.Module):
    def __init__(self, res1, res2):
        super().__init__()
        self.res1 = res1
        self.res2 = res2

    def forward(self, img, target):
        return self.res1(img), self.res2(target)


class DoubleToTensor(torch.nn.Module):
    def __init__(self, t1, t2) -> None:
        super().__init__()
        self.t1 = t1
        self.t2 = t2

    def __call__(self, img, target):
        return self.t1(img), self.t2(target)


class DoubleNormalize(torch.nn.Module):
    def __init__(self, norm1, norm2):
        super().__init__()
        self.norm1 = norm1
        self.norm2 = norm2

    def forward(self, img, target):
        return self.norm1(img), self.norm2(target)


class DoubleRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(target)
        return img, target


class DoubleRandomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(target)
        return img, target


class DoubleRandomApply(T.RandomApply):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p)

    def forward(self, img, target):
        if self.p < torch.rand(1):
            return img, target
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class DoubleRandomRotation(T.RandomRotation):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
        self,
        degrees,
        interpolation=T.InterpolationMode.NEAREST,
        expand=False,
        center=None,
        fill=0,
    ):
        super().__init__(degrees, interpolation, expand, center, fill)

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill1 = self.fill
        fill2 = self.fill
        channels1, _, _ = F.get_dimensions(img)
        channels2, _, _ = F.get_dimensions(target)
        if isinstance(img, torch.Tensor):
            if isinstance(fill1, (int, float)):
                fill1 = [float(fill1)] * channels1
            else:
                fill1 = [float(f) for f in fill1]
        if isinstance(target, torch.Tensor):
            if isinstance(fill2, (int, float)):
                fill2 = [float(fill2)] * channels2
            else:
                fill2 = [float(f) for f in fill2]
        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.interpolation, self.expand, self.center, fill1), F.rotate(
            target, angle, self.interpolation, self.expand, self.center, fill2
        )


def replace_tensor_value_(tensor, a, b):
    tensor[tensor == a] = b
    return tensor


TRANSFORM_MAP = {
    "imagenet": {
        "train": T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    },
    "food11": {
        "train": T.Compose(
            [
                T.Resize((224, 224), interpolation=T.functional.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.25),
                T.RandomVerticalFlip(p=0.25),
                T.RandomApply(
                    transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))],
                    p=0.25,
                ),
                T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=0.25),
                T.RandomApply(
                    transforms=[T.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1)],
                    p=0.25,
                ),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    },
    "cub": {
        "train": T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                AddGaussianNoise(0, 0.05),
                T.RandomHorizontalFlip(),
                T.RandomAffine(10, (0.2, 0.2), (0.8, 1.2)),
                T.Normalize(
                    (0.47473491, 0.48834997, 0.41759949),
                    (0.22798773, 0.22288573, 0.25982403),
                ),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    (0.47473491, 0.48834997, 0.41759949),
                    (0.22798773, 0.22288573, 0.25982403),
                ),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    (0.47473491, 0.48834997, 0.41759949),
                    (0.22798773, 0.22288573, 0.25982403),
                ),
            ]
        ),
    },
    "isic": {
        "train": T.Compose(
            [
                T.Resize((224, 224), interpolation=T.functional.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.25),
                T.RandomVerticalFlip(p=0.25),
                T.RandomApply(
                    transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))],
                    p=0.25,
                ),
                T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=0.25),
                T.RandomApply(
                    transforms=[T.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1)],
                    p=0.25,
                ),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    },
    "mnist": {
        "train": T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        "test": T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
    },
    "splitmnist": {
        "train": T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        "test": T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
    },
    "cifar10": {
        "train": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
            ]
        ),
    },
    "cifar100": {
        "train": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
            ]
        ),
    },
    "splitcifar100": {
        "train": T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    },
    "circles": {
        "train": lambda x: torch.from_numpy(x).float(),
        "test": lambda x: torch.from_numpy(x).float(),
    },
    "blobs": {
        "train": lambda x: torch.from_numpy(x).float(),
        "test": lambda x: torch.from_numpy(x).float(),
    },
    "swirls": {
        "train": lambda x: torch.from_numpy(x).float(),
        "test": lambda x: torch.from_numpy(x).float(),
    },
}


def get_transforms(dataset_name, mode):
    """
    Gets the correct transforms for the dataset
    """

    # Check if dataset_name is supported
    if dataset_name not in TRANSFORM_MAP:
        raise ValueError("Dataset '{}' not supported.".format(dataset_name))

    # Combine transforms
    transforms = TRANSFORM_MAP[dataset_name][mode]

    # Return transforms
    return transforms
