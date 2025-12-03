using BlackBoxOptim

function objective(θ)
    ρ, ϵ, k, λ, T = θ

    # build predictions
    init_matrix = disease_initiation_matrix(
        L,
        amyloid_matrix,
        ones(N),
        ρ, ϵ, k, λ,
        T
    )
    epi = epicenter_accuracy(init_matrix, tau_matrix, 10)
    return -mean(epi)           # minimize → negative to maximize accuracy
end

res = bboptimize(
    objective;
    SearchRange = [
        (0.0, 5.0),   # ρ
        (0.0, 5.0),   # ϵ
        (0.0, 5.0),   # k
        (0.0, 5.0),   # λ
        (0.1, 10.0)   # T
    ],
    NumDimensions = 5,
    MaxTime = 3600,      # run 1 hour
    TraceMode = :silent
)

best_params = best_candidate(res)
best_score  = -best_fitness(res)
