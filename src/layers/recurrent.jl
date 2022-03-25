
mutable struct Recur{T<:AbstractArray}
    state::T
    intial_state::T
end
Recur(state::AbstractArray) = Recur(state, state)
reset!(r::Recur) = r.state = r.intial_state

function (r::Recur)(x, Wx, Wh, b, σ::F) where {F<:Function}
    h = r.state
    h = σ.(Wx*x .+ Wh*h .+ b)
    r.state = h
    return h
end

struct BRNN
    in_size::Int
    out_size::Int # This is the state size
    act::Symbol
end
BRNN(in_size::Int, out_size::Int) = BRNN(in_size, out_size, :tanh)

function create_layer!(ex::Expr, layer::BRNN, layer_num::Int)
    # Adds a Bayesian Recurrent Layer to the model. 
    # We have h = σ(Wₓx + Wₕh + b) recursively
    @unpack in_size, out_size, act = layer

    wx, wh, bias, h0, unit = get_Symbol_names(layer_num, "Wx", "Wh", "b", "h0", "unit")

    # prior specificatoins
    push!(ex.args, :($wx ~ filldist(Normal(), $out_size, $in_size)))
    push!(ex.args, :($wh ~ filldist(Normal(), $out_size, $out_size)))
    push!(ex.args, :($bias ~ filldist(Normal(), $out_size)))
    push!(ex.args, :($h0 ~ filldist(Normal(), $out_size, 1)))

    # creation of the recurrance
    push!(ex.args, :($unit = RNN($act, $wx, $wh, $bias, $h0)))


    return unit, wx, wh, bias, h0
end
