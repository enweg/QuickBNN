
################################################################################
# Constructing BNN
################################################################################

function make_BNN_core(net::ChainBNN, bl::Expr)
    # adding the layers
    quants = Symbol[:sig] 
    units = Symbol[]
    for (i, l) in enumerate(net)
        push!(bl.args, LineNumberNode(i, "Layer"))
        qs = create_layer!(bl, l, i)
        push!(units, qs[1])
        quants = length(qs) > 1 ? vcat(quants, qs[2:end]...) : quants
    end
    push!(bl.args, LineNumberNode(1, "Model Construction"))
    push!(bl.args, Expr(:(=), :net, Expr(:call, :Chain, units...)))
    return :net, quants
end

function make_BNN(net::ChainBNN, likelihood_function; kwargs...)
    model_name = String(Random.shuffle('a':'z')[1:5])
    ex = Meta.parse("@model $model_name(y, x) = begin end")
    bl = ex.args[3].args[2] 
    modelname, quants = make_BNN_core(net, bl)
    push!(bl.args, LineNumberNode(1, "Likelihood"))
    likelihood_function(bl, modelname; kwargs...)
    ret = :(return)
    ret.args[1] = :(()) 
    for q in quants
        push!(ret.args[1].args, :($q = $q))
    end
    push!(bl.args, ret)
    return ex 
end

function BNN(net::ChainBNN, likelihood_function; kwargs...)
    ex = make_BNN(net, likelihood_function; kwargs...)
    nn = Core.eval(@__MODULE__, ex)
    return nn
end

function make_vec(x, sym)
    if length(x) == 1
        return x, sym
    elseif isa(x, AbstractVector)
        ox = vec(x)
        os = [Symbol("$(sym)[$i]") for i in eachindex(x)]
        return ox, os
    else
        ox = vec(x)
        os = Matrix{Symbol}(undef, size(x))
        for j in 1:size(x, 2) 
            for i in 1:size(x, 1)
                os[i,j] = Symbol("$(sym)[$i,$j]")
            end
        end
        return ox, vec(os)
    end
end

function generated_quantities_chain(model, chain)
    chain_params = Turing.MCMCChains.get_sections(chain, :parameters)
    gq = generated_quantities(model, chain_params)
    len = sum([length(v) for v in gq[1,1]])
    vals = Array{Float64}(undef, size(gq,1), len, size(gq,2))
    syms = Array{Symbol}(undef, len)
    for ch in 1:size(gq, 2)
        for draw in 1:size(gq, 1)
            q = gq[draw, ch]
            vs = [make_vec(x, s) for (s, x) in zip(keys(q), q)]
            vals[draw,:,ch] = vcat([v[1] for v in vs]...)
            syms = vcat([s[2] for s in vs]...)
        end
    end
    ch = Chains(vals, syms)
    ch = setinfo(ch, (start_time = zeros(size(vals,3)), stop_time = zeros(size(vals,3))))
    return ch 
end

function posterior_predictive(bnn, x, chain)
    preds = Turing.Inference.predict(bnn(missing, x), chain)
    preds = setinfo(preds, (start_time = fill(0.0, size(chain, 3)), stop_time = fill(0.0, size(chain, 3))))
    return preds
end
