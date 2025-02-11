try:
    import snntorch  # noqa: F401
except ImportError:
    print(
        "The SNN functionality of this package requires extra dependencies ",
        "which can be installed via pip install lfprop[snn] (or lfprop[full] for all dependencies).",
    )
    raise ImportError("snntorch required; reinstall lfprop with option `snn` (pip install lfprop[snn])")


from experiment_utils.training.lightning_models import (
    GradSNNModel,
    OneCycleGradSNNModel,
    OneCycleSNNModel,
    SNNModel,
)


def get_training_method(name):
    methods = {
        "multi_stage_lfp_snn": SNNModel,
        "one_cycle_lfp_snn": OneCycleSNNModel,
        "one_cycle_grad_snn": OneCycleGradSNNModel,
        "multi_stage_grad_snn": GradSNNModel,
    }
    print("\n→ Train Method: {}\n".format(methods[name].__name__))
    return methods[name]
