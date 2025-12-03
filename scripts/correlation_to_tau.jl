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

# make FDG predictions
init_matrix_FDG = disease_initiation_matrix(L, FDG_matrix, ones(N), 1, 1, 1, 1, T)
init_matrix_amy = disease_initiation_matrix(L, amyloid_matrix, ones(N), 1, 1, 1, 1, T)

# compute correlations to tau (FDG)
cors = zeros(size(FDG_matrix,1))
null_cors = zeros(size(FDG_matrix,1))
for i in axes(FDG_matrix,1)
    cors[i] = cor(init_matrix_FDG[i,:], tau_matrix[i,:])
    null_cors[i] = cor(FDG_matrix[i,:], tau_matrix[i,:])
end
histogram(cors, label="Comp Model")
histogram!(null_cors, alpha=0.5, label="Empirical correlation")
savefig("figures/FDG_correlation_to_tau_histogram.png") 
# compute correlations to tau (amyloid)
cors = zeros(size(amyloid_matrix,1))
null_cors = zeros(size(amyloid_matrix,1))
for i in axes(amyloid_matrix,1)
    cors[i] = cor(init_matrix_amy[i,:], tau_matrix[i,:])
    null_cors[i] = cor(amyloid_matrix[i,:], tau_matrix[i,:])
end
histogram(cors, label="Comp Model");
histogram!(null_cors, alpha=0.5, label="Empirical correlation")
savefig("figures/AMY_correlation_to_tau_histogram.png") 

# OPTIMIZE Epicenter hits
using Base.Threads
progress = Atomic{Int}(0)
num_subjects = size(FDG_matrix, 1)

best_params_all = Vector{Tuple{Float64,Float64}}(undef, num_subjects)
best_scores_all = Vector{Float64}(undef, num_subjects)

@threads for i in 1:num_subjects
    function objective_i(θ)
        ρ, ϵ = θ
        init_vector = disease_initiation_vector(
            L,
            diagm(FDG_matrix[i, :]),
            ones(N),
            ρ, ϵ, 1, 1,
            T
        )
        return -cor(init_vector, tau_matrix[i, :])
    end

    res = bboptimize(
        objective_i;
        SearchRange = [(0.0, 20.0), (0.0, 20.0)],
        NumDimensions = 2,
        MaxTime = 600,
        TraceMode = :silent
    )

    best_params_all[i] = best_candidate(res)
    best_scores_all[i] = -best_fitness(res)

    # safe increment
    old = atomic_add!(progress, 1)
    println("Completed $(old+1) / $num_subjects")
end

# Build DataFrame
df = DataFrame(
    subject = 1:num_subjects,
    rho     = [p[1] for p in best_params_all],
    epsilon = [p[2] for p in best_params_all],
    score   = best_scores_all
)

# Save with timestamp
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")
filename = "results/FDG_optimal_parameters_$timestamp.csv"

CSV.write(filename, df)
println("Saved results to $filename")
