__precompile__()
module ml_for_pt

# export package name as MLP
export MLP
const MLP = ml_for_pt

# load packages
using Flux
using StaticArrays
using UnPack
using StatsBase
using DelimitedFiles
using Random
using Base.Threads

# include additional files
include("supervised_learning.jl")
include("learning_by_confusion.jl")
include("prediction_based_method.jl")
include("distributions.jl")
include("utils.jl")

end
