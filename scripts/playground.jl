using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics

# read adjacency matrix from CSV
W = Matrix(CSV.read("data/Schaefer2018_200Parcels_CN.csv", DataFrame; header=false))
N = size(W,1)

# read amyloid data from CSV
df = CSV.read("data/ADNI_A4_cross-sectional_baseline_amy_tau.csv", DataFrame)
#df = df[df.DX .== "CN", :]
amyloid_cols = filter(name -> startswith(name, "centiloid.amyloid.SUVR.Schaefer200"), names(df))
amyloid_matrix = Matrix(df[:, amyloid_cols])
amyloid_matrix = amyloid_matrix ./ median(amyloid_matrix)  # normalize by median
amyloid_matrix = clamp.(amyloid_matrix, 0, Inf)  # threshold at 0.0

# read tau data from CSV
tau_cols = filter(name -> startswith(name, "tau.SUVR.Schaefer200"), names(df))
tau_matrix = Matrix(df[:, tau_cols])

# random Laplacian and activity matrix
L = laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))  # normalize Laplacian
T = 1.0

# make predictions
init_matrix = disease_initiation_matrix(L, amyloid_matrix, ones(N), 1, 1, 1, 1, T)
epi_hits = epicenter_accuracy(init_matrix, tau_matrix, 10)

display("Ratio of epicenter hits: $(mean(epi_hits))")
histogram(epi_hits)



using BlackBoxOptim

function objective(θ)
    ρ, ϵ, k, λ = θ

    # build predictions
    init_matrix = disease_initiation_matrix(
        L,
        amyloid_matrix,
        k/λ*ones(N),
        ρ, ϵ, k, λ,
        T
    )
    epi = epicenter_accuracy(1 ./ init_matrix, tau_matrix, 10)
    return -mean(epi)           # minimize → negative to maximize accuracy
end

res = bboptimize(
    objective;
    SearchRange = [
        (0.0, 10.0),   # ρ
        (0.0, 10.0),   # ϵ
        (0.0, 5.0),   # k
        (0.0, 5.0),   # λ
    ],
    NumDimensions = 4,
    MaxTime = 3600,      # run 1 hour
    TraceMode = :verbose
)

best_params = best_candidate(res)
best_score  = -best_fitness(res)
