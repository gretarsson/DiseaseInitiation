using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics, Dates
using BlackBoxOptim

# read adjacency matrix from CSV
W = Matrix(CSV.read("data/Schaefer2018_200Parcels_CN.csv", DataFrame; header=false))
N = size(W,1)

# load data
FDG_matrix, amyloid_matrix, tau_matrix = nothing, nothing, nothing
#FDG_matrix, amyloid_matrix, tau_matrix = load_dataset(:baseline_amyloid_tau)
FDG_matrix, amyloid_matrix, tau_matrix = load_dataset(:FDG_amyloid_tau_longitudinal)

# drop missing rows
FDG_matrix, amyloid_matrix, tau_matrix = drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)
if FDG_matrix !== nothing
    #FDG_matrix ./= maximum(FDG_matrix)  # normalize by median
    # normalize each row between 0 and 1
    for i in axes(FDG_matrix,1)
        FDG_matrix[i, :] .= (FDG_matrix[i, :] .- minimum(FDG_matrix[i, :])) ./ (maximum(FDG_matrix[i, :]) - minimum(FDG_matrix[i, :]))
    end
end
if amyloid_matrix !== nothing
    #amyloid_matrix ./= maximum(amyloid_matrix)  # normalize by median
    # normalize each row between 0 and 1
    for i in axes(amyloid_matrix,1)
        amyloid_matrix[i, :] .= (amyloid_matrix[i, :] .- minimum(amyloid_matrix[i, :])) ./ (maximum(amyloid_matrix[i, :]) - minimum(amyloid_matrix[i, :]))
    end
end


# random Laplacian and activity matrix
L = laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))  # normalize Laplacian
tspan = (0, 1.0)
Tn = 100
m = 10  # number of epicenter candidates
S = size(tau_matrix,1)

# OPTIMIZE Epicenter hits
# Amyloid:
# WITH k and lambda we get 0.1423 best ratio
# WIITH k,lambda=1 we get 0.141 best ratio 
# FDG
# with k and lambda we get 0.0633 best rato
function objective(θ)
    ϵA, ϵF = θ
    optimal_epis = zeros(S)  # store the optimal epicenter ratio per subject over time, given parameters
    for i in 1:S
        # build predictions
        init_timeseries = disease_initiation_timeseries(
            L,
            diagm(amyloid_matrix[i,:]),
            diagm(FDG_matrix[i,:]),
            1/1*ones(N),
            1, ϵA, ϵF, 0, 0,
            tspan,
            Tn
        )
        epi = epicenter_accuracy(transpose(init_timeseries), tau_matrix, m)  # transpose of timeseries as it's assumed rows are predicitons not columns
        optimal_epis[i] = maximum(epi)
    end
    return -mean(optimal_epis)           # minimize → negative to maximize accuracy
end

res = bboptimize(
    objective;
    SearchRange = [
        (-1.0, 20.0),   # ϵA
        (-1.0, 20.0),   # ϵF
    ],
    NumDimensions = 2,
    MaxTime = 3600,      # run 1 hour
    TraceMode = :verbose
)

best_params = best_candidate(res)
best_score  = -best_fitness(res)

# NULL MODEL (AMYLOID and FDG PET)
amy_pet_epis = epicenter_accuracy(amyloid_matrix, tau_matrix, m)
FDG_pet_epis = epicenter_accuracy(FDG_matrix, tau_matrix, m)

# ================================
# SAVE OPTIMIZATION SUMMARY
# ================================

summary_file = "results/optimal_epicenter_summary_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).txt"

open(summary_file, "w") do io
    println(io, "Optimization summary")
    println(io, "Timestamp: ", Dates.now())
    println(io, "\nBest parameters:")
    println(io, "ϵA = $(best_params[1])")
    println(io, "ϵF = $(best_params[2])")

    println(io, "\nBest score (mean epicenter accuracy): ", best_score)
    println(io, "\nBlackBoxOptim summary:\n")
    println(io, res)

    # ---- NEW: Null model results ----
    println(io, "\nNull model accuracies:")
    println(io, "Amyloid PET null model: $(mean(amy_pet_epis))")
    println(io, "FDG PET null model: $(mean(FDG_pet_epis))")
    # ---------------------------------


end

println("Saved optimization results to $summary_file")



# check optimal solution
optimal_epis = zeros(S)  # store the optimal epicenter ratio per subject over time, given parameters
for i in 1:S
    # build predictions
    init_timeseries = disease_initiation_timeseries(
        L,
        diagm(amyloid_matrix[i,:]),
        diagm(FDG_matrix[i,:]),
        1/1*ones(N),
        1, best_params[1], best_params[2], 0, 0,
        tspan,
        Tn
    )
    epi = epicenter_accuracy(transpose(init_timeseries), tau_matrix, m)  # transpose of timeseries as it's assumed rows are predicitons not columns
    optimal_epis[i] = maximum(epi)
end

histogram(FDG_pet_epis, alpha=0.75, label="FDG PET Null Model, mean = $(round(mean(FDG_pet_epis), digits=2))")
histogram!(amy_pet_epis, alpha=0.75, label="Amyloid PET Null Model, mean = $(round(mean(amy_pet_epis), digits=2))")
histogram!(optimal_epis, title="Optimal Epicenter Accuracy Histogram", xlabel="Epicenter Accuracy", ylabel="Frequency (subjects)", label="Nonlinear model, mean = $(round(mean(optimal_epis), digits=2))", alpha=0.75)  
savefig("figures/optimal_epicenter_histogram_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).png")