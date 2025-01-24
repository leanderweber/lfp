from zennit import layer as zlayer
from zennit import rules as zrules
from zennit import types as ztypes

from ..model import activations

LAYER_MAP_BASE = [
    (ztypes.Activation, zrules.Pass()),
    (activations.Step, zrules.Pass()),
    (zlayer.Sum, zrules.Norm()),
    (ztypes.AvgPool, zrules.Norm()),
]
