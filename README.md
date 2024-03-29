# Replacing neural networks by optimal analytical predictors for the detection of phase transitions
This repository contains a Julia implementation for the approach introduced in our
[paper](https://link.aps.org/doi/10.1103/PhysRevX.12.031044).

### Abstract of the paper
Identifying phase transitions and classifying phases of matter is central to understanding the properties and behavior of a broad range of material systems. In recent years, machine-learning (ML) techniques have been successfully applied to perform such tasks in a data-driven manner. However, the success of this approach notwithstanding, we still lack a clear understanding of ML methods for detecting phase transitions, particularly of those that utilize neural networks (NNs). In this work, we derive analytical expressions for the optimal output of three widely used NN-based methods for detecting phase transitions. These optimal predictions correspond to the results obtained in the limit of high model capacity. Therefore, in practice, they can, for example, be recovered using sufficiently large, well-trained NNs. The inner workings of the considered methods are revealed through the explicit dependence of the optimal output on the input data. By evaluating the analytical expressions, we can identify phase transitions directly from experimentally accessible data without training NNs, which makes this procedure favorable in terms of computation time. Our theoretical results are supported by extensive numerical simulations covering, e.g., topological, quantum, and many-body localization phase transitions. We expect similar analyses to provide a deeper understanding of other classification tasks in condensed matter physics.

![](./assets/method.png)

### This repository

contains code to identify phase transitions from data using supervised learning, learning by confusion, or the prediction-based method, using the optimal analytical expressions (`main_**_opt.jl`) or neural networks (`main_**_NN.jl`). The source files can be found in [source folder](./src/). We provide exemplary code for

* the prototypical probability distributions discussed in our paper, see [the folder](./examples/prototypical_distr/),

* the symmetry-breaking phase transition in the two-dimensional square lattice ferromagnetic Ising model (of size 10 x 10 and 60 x 60), see [the folder](./examples/ising/),

* and the many-body localization phase transition in the Bose-Hubbard chain (of length 6 and 8), see [the folder](./examples/mbl_bose_hubbard/).

The data used to construct the corresponding probability distributions can be found in the [data folder](./data/). Other physical systems can be analyzed in the same fashion by replacing the current probability distributions/data.

### How to run / prerequisites:

- install [julia](https://julialang.org/downloads/)
- download, `activate`, and `instantiate` [`Pkg.instantiate()`] our package
- individual files can then be executed by calling, e.g., `julia main_ising_opt.jl`
- output data/figures are stored in the associated results folder.

## Authors:

- [Julian Arnold](https://github.com/arnoldjulian)
- [Frank Schäfer](https://github.com/frankschae)

```
@article{PhysRevX.12.031044,
  title = {Replacing Neural Networks by Optimal Analytical Predictors for the Detection of Phase Transitions},
  author = {Arnold, Julian and Sch\"afer, Frank},
  journal = {Phys. Rev. X},
  volume = {12},
  issue = {3},
  pages = {031044},
  numpages = {39},
  year = {2022},
  month = {Sep},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevX.12.031044},
  url = {https://link.aps.org/doi/10.1103/PhysRevX.12.031044}
}
```
