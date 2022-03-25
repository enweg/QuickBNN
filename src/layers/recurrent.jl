
import Flux: RNNCell, reshape_cell_output

################################################################################
# RNN 
################################################################################

# The following is equivalent to what Flux has implemented up to a 
# type definition which was taken away to fix a bug with ReverseDiff
function (m::RNNCell{F,A,V,T})(h, x) where {F,A,V,T}
    Wi, Wh, b = m.Wi, m.Wh, m.b
    σ = NNlib.fast_act(m.σ, x)
    h = σ.(Wi*x .+ Wh*h .+ b)
    return h, reshape_cell_output(h, x)
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

    # layer creation
    push!(ex.args, :($unit = RNN($act, $wx, $wh, $bias, $h0)))


    return unit, wx, wh, bias, h0
end

################################################################################
# LSTM
################################################################################
import Flux: multigate, gate

# The following is just adapted code from Flux since 
# raw Flux was not working 

struct LSTMCell{A,V,S}
    Wi::A
    Wh::A
    b::V
    h0::S
    c0::S
end

mutable struct LSTMRecur{T, S}
    cell::T
    h::S
    c::S
end

function (m::LSTMRecur)(x)
    println("Hello Recur")
    m.h, m.c, y = m.cell(m.h, m.c, x)
    println("Bye Recur")
    return y
end

function (m::LSTMCell{A,V,S})(h, c, x) where {A,V,S}
    b, o = m.b, size(h, 1)
    g = m.Wi*x .+ m.Wh*h .+ b
    input, forget, cell, output = multigate(g, o, Val(4))
    c′ = @. sigmoid_fast(forget) * c + sigmoid_fast(input) * tanh_fast(cell)
    h′ = @. sigmoid_fast(output) * tanh_fast(c′)
    return h′, c′, reshape_cell_output(h′, x)
end

LSTM(a...; ka...) = LSTMRecur(LSTMCell(a...; ka...))
LSTMRecur(m::LSTMCell) = LSTMRecur(m, m.h0, m.c0)

# Bayesian LSTM

struct BLSTM
    in_size::Int
    out_size::Int
end

function create_layer!(ex::Expr, layer::BLSTM, layer_num::Int)
    # Adds a Bayesian LSTM layer to the model 
    @unpack in_size, out_size = layer

    wx, wh, bias, h0, unit = get_Symbol_names(layer_num, "Wx", "Wh", "b", "h0", "c0", "unit")

    # prior specifications
    # note that we have four gates, and thus four times as many outputs
    # this follows the flux specifications
    push!(ex.args, :($wx ~ filldist(Normal(), 4*$out_size, $in_size)))
    push!(ex.args, :($wh ~ filldist(Normal(), 4*$out_size, $out_size)))
    push!(ex.args, :($bias ~ filldist(Normal(), 4*$out_size)))
    # push!(ex.args, :($h0 ~ filldist(Normal(), $out_size, 1)))
    push!(ex.args, :($h0 ~ MvNormal(zeros($out_size), I)))
    push!(ex.args, :($h0 = reshape($h0, $out_size, 1)))
    # push!(ex.args, :($c0 ~ filldist(Normal(), $out_size, 1)))

    # Flux LSTM layer
    push!(ex.args, :($unit = LSTM($wx, $wh, $bias, $h0, $h0)))

    return unit, wx, wh, bias, h0
end
