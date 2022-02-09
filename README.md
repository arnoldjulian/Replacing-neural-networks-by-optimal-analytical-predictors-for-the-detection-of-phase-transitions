# Replacing neural networks by optimal analytical predictors for the detection of phase transitions
This repository contains a Julia implementation for the approach introduced in our
[paper](https://arxiv.org/abs/xxxx).

### Abstract of the paper
Identifying phase transitions and classifying phases of matter is central to understanding the properties and behavior of a broad range of material systems. In recent years, machine-learning (ML) techniques have been successfully applied to perform such tasks in a data-driven manner. However, the success of this approach notwithstanding, we still lack a clear understanding of ML methods for detecting phase transitions, particularly of those that utilize neural networks (NNs). In this work, we derive analytical expressions for the optimal output of three widely used NN-based methods for detecting phase transitions. These optimal predictions correspond to the output obtained in the limit of perfectly trained and sufficiently large NNs. The inner workings of the considered methods are revealed through the explicit dependence of the optimal output on the input data. By evaluating the analytical expressions, we can identify phase transitions from data without training NNs. Our theoretical results are supported by extensive numerical simulations covering, e.g., topological, quantum, and many-body localization phase transitions. We expect similar analyses to provide a deeper understanding of other classification tasks in condensed-matter physics.


<p align="center">
  <img src="./misc/method.png" alt="scheme" height="400px" width="748px">
</p>

### Control scenarios

The repository contains different examples for the control of a single qubit:

* SDE control based on full knowledge of the state of the qubit and a continuously
  updated control drive using continuous adjoint sensitivity methods, see [the file](./continuously-updated-control/Control.jl).

* SDE control based on full knowledge of the state of the qubit and a
  piecewise-constant control drive using a direct AD approach, see [the file](./piecewise-constant-control/Control.jl).

* SDE control based on the record of the measured homodyne current and a
  piecewise-constant control drive using a direct AD approach, see [the file](./homodyne-current/Control.jl).

* ODE control (closed quantum system) based on full knowledge of the state of
  the qubit and a continuously updated control drive using continuous adjoint
  sensitivity methods, see [the file](./closed-system/Control.jl).

* SDE control based on full knowledge of the state of the qubit and a continuously
  updated control drive using the hand-crafted strategy, see [the file](./hand_crafted/Control.jl).  


### How to run/ prerequisites:

- install [julia](https://julialang.org/downloads/)
- individual files can be executed by calling, e.g., `julia --threads 10 Control.jl 0.001 1000 1`
  from terminal. Please find the possible parser arguments in the respective julia file.
- output data/figures are stored in the associated data/figures folder.
- other physical systems can be implemented by modifying the respective drift and
  diffusion functions:

  ```julia

  function qubit_drift!(du,u,p,t)
    # expansion coefficients |Ψ> = ce |e> + cd |d>
    ceR, cdR, ceI, cdI = u # real and imaginary parts

    # Δ: atomic frequency
    # Ω: Rabi frequency for field in x direction
    # κ: spontaneous emission
    Δ, Ωmax, κ = p[end-2:end]
    nn_weights = p[1:end-3]
    Ω = (nn(u, nn_weights).*Ωmax)[1]

    @inbounds begin
      du[1] = 1//2*(ceI*Δ-ceR*κ+cdI*Ω)
      du[2] = -cdI*Δ/2 + 1*ceR*(cdI*ceI+cdR*ceR)*κ+ceI*Ω/2
      du[3] = 1//2*(-ceR*Δ-ceI*κ-cdR*Ω)
      du[4] = cdR*Δ/2 + 1*ceI*(cdI*ceI+cdR*ceR)*κ-ceR*Ω/2
    end
    return nothing
  end

  function qubit_diffusion!(du,u,p,t)
    ceR, cdR, ceI, cdI = u # real and imaginary parts

    @inbounds begin
      du[2] += sqrt(κ)*ceR
      du[4] += sqrt(κ)*ceI
    end
    return nothing
  end

  ```
  to the system at-hand.

- alternative basis expansions replacing the neural networks are described in
  the [docs](https://diffeqflux.sciml.ai/dev/layers/BasisLayers/). For instance,
  one may use a tensor layer of a polynomial basis expansion:
  ```julia
  A = [PolynomialBasis(5)]
  nn = TensorLayer(A, 4)
  p_nn = nn.p
  ```


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
