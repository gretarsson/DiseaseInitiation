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
FDG_matrix, amyloid_matrix, tau_matrix = load_dataset(:FDG_amyloid_tau_longitudinal; DX=nothing)

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
init_matrix_FDG = disease_initiation_matrix(L, FDG_matrix, ones(N), 1, 5, 1, 1, T)
init_matrix_amy = disease_initiation_matrix(L, amyloid_matrix, ones(N), 5, 1, 1, 1, T)

# find the rank of the highest predicted region(s)
S = size(FDG_matrix, 1)
ranks_FDG = Vector{Int}(undef, S)
ranks_amy = Vector{Int}(undef, S)
for i in 1:S
    # epicenter per the model prediction
    epi_FDG = argmax(init_matrix_FDG[i, :]) 
    epi_amy = argmax(init_matrix_amy[i, :]) 

    # find epicenter rank in empirical data
    ranks_FDG[i] = sortperm(tau_matrix[i,:]; rev=true) |> x -> findfirst(==(epi_FDG), x)
    ranks_amy[i] = sortperm(tau_matrix[i,:]; rev=true) |> x -> findfirst(==(epi_amy), x)
end
mean(ranks_FDG)


# shuffle predictions
null_ranks_FDG = Vector{Int}(undef, S)
null_ranks_amy = Vector{Int}(undef, S)
for i in 1:S
    # epicenter per the model prediction
    epi_FDG = argmax(shuffle(init_matrix_FDG[i, :]))
    epi_amy = argmax(shuffle(init_matrix_amy[i, :]))

    # find epicenter rank in empirical data
    null_ranks_FDG[i] = sortperm(tau_matrix[i,:]; rev=true) |> x -> findfirst(==(epi_FDG), x)
    null_ranks_amy[i] = sortperm(tau_matrix[i,:]; rev=true) |> x -> findfirst(==(epi_amy), x)
end

using Plots

bins = 1:200   # ranks range 1..N

##########################
# FDG histogram
##########################
p_FDG = histogram(ranks_FDG;
    bins=bins,
    normalize=:pdf,
    label="Model",
    xlabel="Rank (1 = best)",
    ylabel="Density",
    title="FDG Rank Distribution",
    alpha=0.6,
);
histogram!(null_ranks_FDG;
    bins=bins,
    normalize=:pdf,
    label="Shuffled",
    alpha=0.6,
);

savefig(p_FDG, "figures/FDG_rank_histogram.png")


##########################
# Amyloid histogram
##########################
p_amy = histogram(ranks_amy;
    bins=bins,
    normalize=:pdf,
    label="Model",
    xlabel="Rank (1 = best)",
    ylabel="Density",
    title="Amyloid Rank Distribution",
    alpha=0.6,
);
histogram!(null_ranks_amy;
    bins=bins,
    normalize=:pdf,
    label="Shuffled",
    alpha=0.6,
);

savefig(p_amy, "figures/amyloid_rank_histogram.png")
