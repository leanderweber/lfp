from zennit import rules as zrules
from zennit import types as ztypes
from zennit import layer as zlayer

from ..model import activations

LAYER_MAP_BASE = [
    (ztypes.Activation, zrules.Pass()),
    (activations.Step, zrules.Pass()),
    (activations.NegStep, zrules.Pass()),
    (activations.Linearact, zrules.Pass()),
    (activations.StepReLU, zrules.Pass()),
    (activations.StepLeakyReLU, zrules.Pass()),
    (zlayer.Sum, zrules.Norm()),
    (ztypes.AvgPool, zrules.Norm()),
]
