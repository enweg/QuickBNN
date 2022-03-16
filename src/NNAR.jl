
################################################################################
# Simulation and Construction of non-linear AR(p) using NN 
################################################################################

struct NNAR{C<:Flux.Chain, D<:Distribution, R<:Random.AbstractRNG}
    chain::C
    burnin::Int
    e_dist::D
    rng::R
    seed::Int
end

# Note that the normal is parameterised by the SD not VAR
NNAR(chain; burnin=1000, e_dist = Normal(0, 1), rng = Random.GLOBAL_RNG) = NNAR(chain, burnin, e_dist, rng, rng == Random.GLOBAL_RNG ? 0 : Int(rng.seed[1]))

function (mod::NNAR)(;N=100, starting_values = zeros(size(mod.chain[1].weight)[2]), reseed = false)
    if reseed
        if (mod.rng == Random.GLOBAL_RNG) error("Cannot reset seed for global RNG") end
        Random.seed!(mod.rng, mod.seed) 
    end
    p = size(mod.chain[1].weight)[2] 
    iters = N + p + mod.burnin
    y = zeros(iters)
    y[1:p] = starting_values 
    for n=(p+1):iters
        y[n] = mod.chain(y[(n-p):(n-1)])[1] + rand(mod.rng, mod.e_dist)
    end
    return y[(end-N+1):end] 
end

function theoretical_dist(mod::NNAR, data::Vector{Float64})
    # This return the theoretical distribution at each point t given 
    # information up until time t-1 
    # note that the theoretical distribution can 
    # only be provided for observations from p+1 onwards
    p = size(mod.chain[1].weight)[2] 
    mu = zeros(length(data)-p)
    v = zeros(length(data)-p)
    for t=(p+1):length(data)
        mu[t-p] = mod.chain(data[t-p:t-1])[1]
        v[t-p] = var(mod.e_dist)
    end
    return mu, v
end
