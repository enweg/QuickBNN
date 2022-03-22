
################################################################################
# Constructing BNN
################################################################################

function make_BNN(net::ChainBNN; link::Symbol = :Normal, ν = missing)
    model_name = String(Random.shuffle('a':'z')[1:5])
    ex = Meta.parse("@model $model_name(y, x) = begin end")
    bl = ex.args[3].args[2] 
    # the variance
    push!(bl.args, :(sig ~ InverseGamma())) 
    # adding the layers
    quants = Symbol[:sig] 
    input = :x
    for (i, l) in enumerate(net)
        qs = create_layer!(bl, l, i, input)
        input = qs[1]
        quants = length(qs) > 1 ? vcat(quants, qs[2:end]...) : quants
    end
    allowed_links = [:Normal, :TDist]
    if link == :Normal
        push!(bl.args, :(y ~ MvNormal(vec($input), sig*I)))
    elseif link == :TDist 
        if (ismissing(ν)) @warn("No ν (df of TDist) provided. Defaulting to 30"); ν = 30 end
        push!(bl.args, :(Turing.@addlogprob!(sum(logpdf.(TDist($ν), (y.-vec($input)).-sig)./sig))))
    else
        @error("$([link]) linkfunction is not implemented. Currently implemented are $allowed_links")
    end
    ret = :(return)
    ret.args[1] = :(()) 
    for q in quants
        push!(ret.args[1].args, :($q = $q))
    end
    push!(bl.args, ret)
    return ex
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

function BNN(net::ChainBNN; kwargs...)
    ex = make_BNN(net; kwargs...)
    nn = Core.eval(@__MODULE__, ex)
    return nn
end