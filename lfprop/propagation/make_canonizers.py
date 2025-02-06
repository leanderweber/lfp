import torchvision
from zennit import canonizers
from zennit import torchvision as zvision


def get_zennit_canonizer(model):
    """
    Checks the type of model and selects the corresponding zennit canonizer
    """

    # ResNet
    if isinstance(model, torchvision.models.ResNet):
        return zvision.ResNetCanonizer

    # VGG
    if isinstance(model, torchvision.models.VGG):
        return zvision.VGGCanonizer

    # default fallback (only the above types have specific canonizers in zennit for now)
    return canonizers.SequentialMergeBatchNorm
