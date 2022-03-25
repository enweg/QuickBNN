module QuickBNN

################################################################################
# Dependencies
################################################################################
using Turing 
using Flux
using LinearAlgebra
using Parameters
using MacroTools: @forward
using Random
using Distributions
using Bijectors

################################################################################
# Includes 
################################################################################
include("utils.jl")
include("NNAR.jl")
include("AR.jl")
include("./layers/dense.jl")
include("./layers/recurrent.jl")
include("model.jl")
include("./likelihoods.jl")

################################################################################
# Exports
################################################################################
export BDense, DenseOrderedBias, DenseOrderedWeights, DenseForcePosFirstWeight, ChainBNN
export BRNN
export NNAR, theoretical_dist
export make_BNN, generated_quantities_chain, posterior_predictive, BNN
export AR, theoretical_dist
export to_RNN_format, reset!

end # module
