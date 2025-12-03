module DiseaseInitiation

export laplacian, disease_initiation_eqn, disease_initiation_matrix, epicenter_accuracy

using DifferentialEquations, LinearAlgebra
function make_disease_initiation(L, M)
    function f!(du, u, p, t)
        ρ, ϵ, k, λ = p
        du .= -ρ * L * (I + ϵ .* M) * u .+ k .- λ .* u
    end
    return f!
end

function laplacian(A; kind::Symbol = :out)
    size(A,1) == size(A,2) ||
        throw(ArgumentError("Adjacency matrix must be square"))

    if kind == :out
        d = sum(A, dims=2)            # row sums
    elseif kind == :in
        d = sum(A, dims=1)'           # column sums
    else
        throw(ArgumentError("kind must be :out or :in"))
    end

    D = Diagonal(vec(d))
    return D - A
end
function disease_initiation_eqn(L, M, u0, ρ, ϵ, k, λ, T)
    N = length(u0)

    # Base linear operator X
    X = -ρ * L * (I + ϵ*M) - λ * I

    # Constant forcing term b = k * ones
    b = k .* ones(N)

    # Build augmented matrix
    Xtilde = zeros(N+1, N+1)
    Xtilde[1:N, 1:N] = X
    Xtilde[1:N, N+1] = b
    # last row remains zeros (dummy variable)

    # Augmented initial condition
    utilde0 = vcat(u0, 1.0)

    # Solve via matrix exponential
    utilde_T = exp(Xtilde * T) * utilde0

    # Return only physical variables (discard dummy)
    return utilde_T[1:N]
end


function disease_initiation_matrix(L, M_matrix, u0, ρ, ϵ, k, λ, T)
    N = size(M_matrix,2)
    S = size(M_matrix,1)
    sol_matrix = zeros(S,N)
    for i in axes(M_matrix,1)
        M = diagm(M_matrix[i,:])
        sol_matrix[i,:] = disease_initiation_eqn(L, M, u0, ρ, ϵ, k, λ, T)
    end
    return sol_matrix
end


function epicenter_accuracy(pred_matrix, tau_matrix, k)
    hits = zeros(size(pred_matrix,1))
    for i in axes(pred_matrix,1)
        epi_pred = sortperm(pred_matrix[i,:], rev=true)[1:k]
        highest_tau_inds = sortperm(tau_matrix[i,:], rev=true)[1:k]
        hits[i] = length(intersect(epi_pred, highest_tau_inds))/k  
    end
    return hits
end

end