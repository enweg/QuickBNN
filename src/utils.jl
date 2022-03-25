################################################################################
# helper function 
################################################################################

get_Flux_fast_act(act::Symbol) = Symbol(Flux.NNlib.fast_act(eval(act)))

get_Symbol_names(layer_num, args...) = [Symbol("$(n)$(layer_num)") for n in args]

get_symbol_from_var(var) = eval(:(Symbol("$($var)")))

function to_RNN_format(sequences::Vector{Matrix{T}}) where {T}
    # Assumes that each matrix is of the format n_features x len_seq 
    # We need a vector of matrices of format n_featurs x n_seq for RNN layers 
    n_seq = length(sequences)
    n_features, len_seq = size(sequences[1])
    out = [ones(n_features, n_seq) for _ in 1:len_seq]
    for t=1:len_seq 
        out[t] = hcat([seq[:,t] for seq in sequences]...)
    end
    return out
end