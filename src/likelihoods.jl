
# Standard feed forward
function likelihood_normal(bl::Expr, modelname::Symbol)
    push!(bl.args, :(sig ~ InverseGamma()))
    push!(bl.args, :(y ~ MvNormal(vec($modelname(x)), sig*I)))
end

# sequence to sequence
# assumes that x is a vector of matrices and only the last output is used 
# matrices: n_features x n_seq
# vector: of length len_seq
function likelihood_normal_seq_to_one(bl::Expr, modelname::Symbol)
    push!(bl.args, :(sig ~ InverseGamma()))
    push!(bl.args, :(y ~ MvNormal(vec([$modelname(xx) for xx in x][end]), sig*I)))
end

# standard feed forward
function likelihood_tdist(bl::Expr, modelname::Symbol; ν=missing)
    if (ismissing(ν)) @warn("No ν (df of TDist) provided. Defaulting to ν=30"); ν = 30 end
    push!(bl.args, :(sig ~ InverseGamma()))
    push!(bl.args, :(Turing.@addlogprob!(sum(logpdf.(TDist($ν), (y.-vec($modelname(x))).-sig)./sig))))
end

# sequence to sequence
# assumes that x is a vector of matrices and only the last output is used 
# matrices: n_features x n_seq
# vector: of length len_seq
function likelihood_tdist_seq_to_one(bl::Expr, modelname::Symbol; ν=missing)
    if (ismissing(ν)) @warn("No ν (df of TDist) provided. Defaulting to ν=30"); ν = 30 end
    push!(bl.args, :(sig ~ InverseGamma()))
    push!(bl.args, :(Turing.@addlogprob!(sum(logpdf.(TDist($ν), (y.-vec([$modelname(xx) for xx in x][end])).-sig)./sig))))
end