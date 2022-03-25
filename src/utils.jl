################################################################################
# helper function 
################################################################################

get_Flux_fast_act(act::Symbol) = Symbol(Flux.NNlib.fast_act(eval(act)))

get_Symbol_names(layer_num, args...) = [Symbol("$(n)$(layer_num)") for n in args]

get_symbol_from_var(var) = eval(:(Symbol("$($var)")))

# Transforms a vector of matrices to a tensor which can then be fed into 
# an RNN. This allows handling of multiple sequences at once
# tensor will be of dimensions n_features x n_seq x len_seq 
function to_RNN_tensor(sequences::Vector{Matrix{T}}) where {T <: Real}
    # T = eltype(sequences[1])
    n_seq = length(sequences)
    n_features, len_seq = size(sequences[1])
    tensor = zeros(T, n_features, n_seq, len_seq)
    for (i, seq) in enumerate(sequences)
        tensor[:,i,:] = seq
    end
    return tensor
end
