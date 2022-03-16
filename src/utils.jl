################################################################################
# helper function 
################################################################################

get_Flux_fast_act(act::Symbol) = Symbol(Flux.NNlib.fast_act(eval(act)))

get_Symbol_names(layer_num, args...) = [Symbol("$(n)$(layer_num)") for n in args]

get_symbol_from_var(var) = eval(:(Symbol("$($var)")))