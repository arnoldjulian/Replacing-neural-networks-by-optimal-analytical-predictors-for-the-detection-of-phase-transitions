# Replacing neural networks by optimal analytical predictors for the detection of phase transitions
This repository contains a Julia implementation for the approach introduced in our
[paper](https://arxiv.org/abs/xxxx).

### Abstract of the paper
Identifying phase transitions and classifying phases of matter is central to understanding the properties and behavior of a broad range of material systems. In recent years, machine-learning (ML) techniques have been successfully applied to perform such tasks in a data-driven manner. However, the success of this approach notwithstanding, we still lack a clear understanding of ML methods for detecting phase transitions, particularly of those that utilize neural networks (NNs). In this work, we derive analytical expressions for the optimal output of three widely used NN-based methods for detecting phase transitions. These optimal predictions correspond to the output obtained in the limit of perfectly trained and sufficiently large NNs. The inner workings of the considered methods are revealed through the explicit dependence of the optimal output on the input data. By evaluating the analytical expressions, we can identify phase transitions from data without training NNs. Our theoretical results are supported by extensive numerical simulations covering, e.g., topological, quantum, and many-body localization phase transitions. We expect similar analyses to provide a deeper understanding of other classification tasks in condensed-matter physics.

![](./misc/method.png)

### This repository

contains code to identify phase transitions from data using supervised learning, learning by confusion, or the prediction-based method, using the optimal analytical expressions or neural networks. The source files can be found in [source folder](./src/). We provide exemplary code for

* the prototypical probability distributions discussed in our paper, see [the folder](./examples/prototypical_distr/),

* the symmetry-breaking phase transition in the two-dimensional square lattice ferromagnetic Ising model (of size 10 x 10 and 60 x 60), see [the folder](./examples/ising/),

* and the many-body localization phase transition in the Bose-Hubbard chain (of length 6 and 8), see [the folder](./examples/mbl_bose_hubbard/).

The data used to construct the corresponding probability distributions can be found in the data folder. Other physical systems can be analyzed in the same fashion by replacing the current probability distributions/data.

### How to run / prerequisites:

- install [julia](https://julialang.org/downloads/)
- individual files can be executed by calling, e.g., `julia main_ising_opt.jl`
- output data/figures are stored in the associated save folder.

## Authors:

- [Julian Arnold](https://github.com/arnoldjulian)
- [Frank Schäfer](https://github.com/frankschae)

```
@article{arnold:2022,
  title={Replacing neural networks by optimal analytical predictors for the detection of phase transitions},
  author={Arnold, Julian and Sch\"{a}fer, Frank},
  journal={arXiv preprint arXiv:xxx},
  year={2022}
}
```
