import torch
import torchvision

from lfprop.model import activations, custom_resnet

from . import model_definitions

TORCHMODEL_MAP = {
    "vgg16": torchvision.models.vgg16,
    "resnet18": custom_resnet.custom_resnet18,
    "resnet34": custom_resnet.custom_resnet34,
    "customresnet18": custom_resnet.custom_resnet18,
    "vgg16bn": torchvision.models.vgg16_bn,
}

MODEL_MAP = {
    "lenet": model_definitions.LeNet,
    "cifar-vgglike": model_definitions.CifarVGGLike,
    "cifar-vgglike-bn": model_definitions.CifarVGGLikeBN,
    "dense-only": model_definitions.DenseOnly,
    "toydata-dense": model_definitions.ToyDataDense,
}

ACTIVATION_MAP = {
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "silu": torch.nn.SiLU,
    "leakyrelu": torch.nn.LeakyReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "step": activations.Step,
}

EXCLUDED_MODULE_TYPES = [
    torchvision.models.VGG,
    model_definitions.LeNet,
    model_definitions.CifarVGGLike,
    model_definitions.DenseOnly,
]


def normal_pos(tensor, *args, **kwargs):
    tensor.data = tensor.data.abs()


def normal_neg(tensor, *args, **kwargs):
    tensor.data = -tensor.data.abs()


INIT_FUNCS = {"positive": normal_pos, "negative": normal_neg}


def replace_torchvision_last_layer(model, n_outputs):
    if isinstance(model, torchvision.models.VGG) or isinstance(model, torchvision.models.efficientnet.EfficientNet):
        classifier = model.classifier
        modules = [m for m in classifier.modules() if not isinstance(m, torch.nn.Sequential)]
        modules[-1] = torch.nn.Linear(modules[-1].in_features, n_outputs)
        model.classifier = torch.nn.Sequential(*modules)
    elif isinstance(model, torchvision.models.ResNet) or isinstance(model, torchvision.models.Inception3):
        classifier = model.fc
        model.fc = torch.nn.Linear(classifier.in_features, n_outputs)
    else:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, n_outputs)


def replace_torchvision_activations(model, activation):
    if isinstance(model, torchvision.models.VGG):
        for module in model.modules():
            if isinstance(module, torch.nn.Sequential):
                seq_modules = [m for m in module.modules() if not isinstance(m, torch.nn.Sequential)]
                for i, mod in enumerate(seq_modules):
                    if isinstance(mod, torch.nn.ReLU):
                        module[i] = activation()
        if len([m for m in model.modules() if isinstance(m, torch.nn.ReLU)]) > 0:
            raise ValueError("There are still ReLUs left after replacement!")
    elif isinstance(model, torchvision.models.ResNet):
        for module in model.modules():
            if hasattr(module, "relu"):
                module.relu = activation()
        if len([m for m in model.modules() if isinstance(m, torch.nn.ReLU)]) > 0:
            raise ValueError("There are still ReLUs left after replacement!")
    else:
        print("Model type not supported, not changing activations")


def init_model_weights(model, init_func):
    def param_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
            init_func(m.weight)
            if m.bias is not None:
                init_func(m.bias)

    model.apply(param_init)


def get_model(model_name, n_channels, n_outputs, device, **kwargs):
    """
    Gets the correct model
    """

    replace_last_layer = kwargs.get("replace_last_layer", True)

    # Check if model_name is supported
    if model_name not in MODEL_MAP and model_name not in TORCHMODEL_MAP:
        raise ValueError("Model '{}' is not supported.".format(model_name))

    # Build model
    activation = kwargs.get("activation", "relu")
    if model_name in MODEL_MAP:
        model = MODEL_MAP[model_name](
            n_channels=n_channels,
            n_outputs=n_outputs,
            activation=ACTIVATION_MAP[activation],
        )
    elif model_name in TORCHMODEL_MAP:
        model = TORCHMODEL_MAP[model_name](pretrained=kwargs.get("pretrained_model", True))
        if replace_last_layer:
            replace_torchvision_last_layer(model, n_outputs)
        if activation != "relu":
            replace_torchvision_activations(model, ACTIVATION_MAP[activation])

    if "init_func" in kwargs.keys() and kwargs.get("init_func") != "default":
        init_func = INIT_FUNCS[kwargs.get("init_func")]
        init_model_weights(model, init_func)

    # Return model on correct device
    return model.to(device)


def list_layers(model):
    """
    List module layers
    """

    # Exclude specific types of modules
    layers = [module for module in model.modules() if type(module) not in [torch.nn.Sequential] + EXCLUDED_MODULE_TYPES]

    return layers
