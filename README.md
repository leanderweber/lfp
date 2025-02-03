<div align="center">
  <img src="docs/source/_static/lfp_logo.png" width="300"/>
  <p>Gradient-free Neural Network Training based on Layer-wise Relevance Propagation (LRP)</p>
</div>

### :octopus: Flexibility
LFP is highly flexible w.r.t. the models and objective functions it can be used with, as it does not require differentiability. 
Consequently, it can be applied in non-differentiable architectures (e.g., Spiking Neural Networks) without requiring further adaptations, 
and naturally handles discrete objectives, such as feedback directly obtained from humans.

### :gear: Efficiency
LFP applies an implicit weight-scaling of updates and only propagates feedback through nonzero connections and activations. This leads to sparsity of updates and the final model, while not sacrificing performance or convergence speed meaningfully compared to gradient descent. The obtained models can be pruned more easily since they represent information more efficiently.

### :open_book: Paper
For more details, refer to our paper 
TODO: Add Link & Citation

### :scroll: License
This project is licensed under the BSD-3 Clause License, since LRP (which LFP is based on) is a patented technology that can only be used free of charge for personal and scientific purposes.

## :rocket: Getting Started


### :fire: Installation

LFP is available from PyPI, to install simply run

```shell
$ pip install lfprop
```

If you would like to check out the ```minimal example.ipynb``` notebook, run 

```shell
$ pip install lfprop[quickstart]
```

instead to install the necessary dependencies.

If you would like to run the scripts and notebooks for reproducing the paper experiments, you can run

```shell
$ pip install lfprop[full]
```

instead to install the full dependencies.

### Overview



### Examples


### :mag: Reproducing Experiments

To reproduce experiments from the paper


## :pencil2: Contributing
