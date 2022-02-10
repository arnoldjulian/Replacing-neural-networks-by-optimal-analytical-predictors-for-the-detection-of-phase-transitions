__precompile__()
module my_ml_for_pt

export MLP
const MLP = my_ml_for_pt

# load packages
using Flux
using StaticArrays
using UnPack
using StatsBase
using DelimitedFiles
using Random
using Base.Threads

# using LinearAlgebra
# using Plots, DelimitedFiles, LaTeXStrings
# using DiffEqFlux, Flux, Zygote
# using Random

include("supervised_learning.jl")
include("learning_by_confusion.jl")
include("prediction_based_method.jl")
include("distributions.jl")
include("utils.jl")

end
