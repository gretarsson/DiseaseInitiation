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
    FDG_matrix ./= median(FDG_matrix)  # normalize by median
end
if amyloid_matrix !== nothing
    amyloid_matrix ./= median(amyloid_matrix)  # normalize by median
end


# random Laplacian and activity matrix
L = laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))  # normalize Laplacian
T = 1.0
m = 10  # number of epicenter candidates

# OPTIMIZE Epicenter hits
# Amyloid:
# WITH k and lambda we get 0.1423 best ratio
# WIITH k,lambda=1 we get 0.141 best ratio 
# FDG
# with k and lambda we get 0.0633 best rato
function objective(θ)
    ρ, ϵ, k, λ = θ
    # build predictions
    init_matrix = disease_initiation_matrix(
        L,
        FDG_matrix,
        k/λ*ones(N),
        ρ, ϵ, k, λ,
        T
    )
    epi = epicenter_accuracy(init_matrix, tau_matrix, m)
    return -mean(epi)           # minimize → negative to maximize accuracy
end

res = bboptimize(
    objective;
    SearchRange = [
        (0.0, 20.0),   # ρ
        (0.0, 20.0),   # ϵ
        (0.0, 20.0),   # k
        (0.0, 20.0),   # λ
    ],
    NumDimensions = 4,
    MaxTime = 3600,      # run 1 hour
    TraceMode = :verbose
)

best_params = best_candidate(res)
best_score  = -best_fitness(res)

# ================================
# SAVE OPTIMIZATION SUMMARY
# ================================

summary_file = "results/optimization_summary_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).txt"

open(summary_file, "w") do io
    println(io, "Optimization summary")
    println(io, "Timestamp: ", Dates.now())
    println(io, "\nBest parameters:")
    println(io, "ρ = $(best_params[1])")
    println(io, "ϵ = $(best_params[2])")
    println(io, "k = $(best_params[3])")
    println(io, "λ = $(best_params[4])")

    println(io, "\nBest score (mean epicenter accuracy): ", best_score)
    println(io, "\nBlackBoxOptim summary:\n")
    println(io, res)
end

println("Saved optimization results to $summary_file")
