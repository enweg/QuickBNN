
################################################################################
# Layers 
################################################################################

struct DenseBNN
    in_size::Int
    out_size::Int
    act::Symbol
end
DenseBNN(in_size::Int, out_size::Int) = DenseBNN(in_size, out_size, :identity)

struct DenseOrderedBias
    in_size::Int
    out_size::Int
    act::Symbol
end
DenseOrderedBias(in_size::Int, out_size::Int) = DenseOrderedBias(in_size, out_size, :identity)

struct DenseOrderedWeights
    in_size::Int
    out_size::Int
    act::Symbol
end
DenseOrderedWeights(in_size::Int, out_size::Int) = DenseOrderedWeights(in_size, out_size, :identity)

struct DenseForcePosFirstWeight
    in_size::Int
    out_size::Int
    act::Symbol
end
DenseForcePosFirstWeight(in_size::Int, out_size::Int) = DenseForcePosFirstWeight(in_size, out_size, :identity)

struct ChainBNN{T<:Union{Tuple, NamedTuple, AbstractVector}}
    layers::T
end

ChainBNN(args...) = ChainBNN(args)

@forward ChainBNN.layers Base.length, Base.getindex, Base.iterate, 
        Base.first, Base.last, Base.lastindex

################################################################################
# create layer implementations
################################################################################

function create_layer!(ex::Expr, layer::DenseBNN, layer_num::Int, input::Symbol)
    # Adds a dense layer to a turing model by adding it to ex. 
    # output variable symbol will be returned for later use. 
    @unpack in_size, out_size, act = layer
    act = get_Flux_fast_act(act)

    weight, bias, out = get_Symbol_names(layer_num, "W", "b", "output")

    push!(ex.args, :($weight ~ filldist(Normal(), $out_size, $in_size)))
    push!(ex.args, :($bias ~ filldist(Normal(), $out_size)))
    push!(ex.args, :($out = $act.($weight * $input .+ $bias)))

    return out, weight, bias
end

function create_layer!(ex::Expr, layer::DenseOrderedBias, layer_num::Int, input::Symbol)
    # Same as dense layer but forces the biases to be ordered from 
    # smallest to largest by putting a truncated normal prior on all biases
    # except the first one and definining biases b1*=b1, b2*=b1* + b2 ...
    @unpack in_size, out_size, act = layer
    if (out_size == 1) error("DenseOrderedBias layers are only suitable for more than one output") end
    act = get_Flux_fast_act(act)

    weight, bfirst, binc, bias, out = get_Symbol_names(layer_num, "W", "bfrst", "binc", "b", "output")

    push!(ex.args, :($weight ~ filldist(Normal(), $out_size, $in_size)))
    push!(ex.args, :($bfirst ~ Normal()))
    push!(ex.args, :($binc ~ filldist(TruncatedNormal(0, 1, 0, Inf), $(out_size - 1))))
    push!(ex.args, :($bias = cumsum(vcat($bfirst, $binc))))
    push!(ex.args, :($out = $act.($weight * $input .+ $bias)))

    return out, weight, bias
end

function create_layer!(ex::Expr, layer::DenseOrderedWeights, layer_num::Int, input::Symbol)
    # Similar to dense but forces an ordering of the weights by ordering the first
    # column of the weight matrix. This is done by setting for w[1,1] a normal prior and for
    # w[*,1] a truncated normal and defninint w*[1,1] = w[1,1], w*[i, 1] = w*[i-1, 1]
    # The ramaining columns of the weight matrix have a normal prior
    @unpack in_size, out_size, act = layer
    if (out_size == 1) error("DenseOrderedWeights layers are only suitable for more than one output") end
    act = get_Flux_fast_act(act)

    w1first, w1inc, wright, weight, bias, out = get_Symbol_names(layer_num, "wfirst", "winc", "wright", "W", "b", "output")

    push!(ex.args, :($w1first ~ Normal()))
    push!(ex.args, :($w1inc ~ filldist(TruncatedNormal(0, 1, 0, Inf), $(out_size - 1))))
    if (in_size > 1)
        push!(ex.args, :($wright = filldist(Normal(), $out_size, $(in_size-1))))
        push!(ex.args, :($weight = hcat(cumsum(vcat($w1first, $w1inc)), $w1right)))
    else
        push!(ex.args, :($weight = reshape(cumsum(vcat($w1first, $w1inc)), $out_size, $in_size)))
    end

    push!(ex.args, :($bias ~ filldist(Normal(), $out_size)))
    push!(ex.args, :($out = $act.($weight * $input .+ $bias)))

    return out, weight, bias
end

function create_layer!(ex::Expr, layer::DenseForcePosFirstWeight, layer_num::Int, input::Symbol)
    # Forces the first weight in the first column of the weight matrix to be 
    # positive. This is done by putting a truncated normal prior on it
    @unpack in_size, out_size, act = layer
    act = get_Flux_fast_act(act)

    wpos, wrest, weight, bias, out = get_Symbol_names(layer_num, "wpos", "Wrest", "W", "b", "output")

    push!(ex.args, :($wpos ~ TruncatedNormal(0, 1, 0, Inf)))
    if out_size == 1
        push!(ex.args, :($weight = $wpos))
    else
        push!(ex.args, :($wrest ~ filldist(Normal(), $(in_size * out_size - 1))))
        push!(ex.args, :($weight = reshape(vcat($wpos, $wrest), $out_size, $in_size)))
    end
    push!(ex.args, :($bias ~ Normal($out_size)))
    push!(ex.args, :($out = $act.($weight * $input .+ $bias)))

    return out, weight, bias
end
