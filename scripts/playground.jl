using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics, Dates
default(dpi=300)

# read adjacency matrix from CSV
W = Matrix(CSV.read("data/Schaefer2018_200Parcels_CN.csv", DataFrame; header=false))
N = size(W,1)

# load data
println("\nloading dataset...")
FDG_matrix, amyloid_matrix, tau_matrix = nothing, nothing, nothing
FDG_matrix, amyloid_matrix, tau_matrix, subject_IDs = load_dataset(:FDG_amyloid_tau_longitudinal; centiloid_threshold=nothing)
println("...using $(size(FDG_matrix,1)) subjects.\n")

# drop missing rows and normalize
FDG_matrix, amyloid_matrix, tau_matrix, nonmissing_subj = drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)
FDG_matrix = normalize_rows(FDG_matrix)
amyloid_matrix = normalize_rows(amyloid_matrix)
subject_IDs = subject_IDs[nonmissing_subj]  # keep track of subject IDs after dropping missing


# random Laplacian and activity matrix
L = laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))  # normalize Laplacian
tspan = (0,20.0)
Tn = 100

# make predictions
#init_timeseries = disease_initiation_timeseries(L, diagm(vec(mean(amyloid_matrix;dims=1))), diagm(vec(mean(FDG_matrix;dims=1))), ones(N), 1, 1, 1, 0, 0, tspan, Tn)
for i in axes(amyloid_matrix,1)
    amyloid_M = diagm(amyloid_matrix[i,:])
    FDG_M = diagm(FDG_matrix[i,:])
    init_timeseries = disease_initiation_timeseries(L, amyloid_M, FDG_M, ones(N), 1, 0, 1, 0, 0, tspan, Tn)

    # plot timeseries
    #t = range(tspan[1], tspan[2], length=Tn)
    #plot(t, init_timeseries',  # transpose so each trajectory is a series
    #    xlabel = "Time",
    #    ylabel = "u(t)",
    #    title = "All regions",
    #    lw = 2,
    #    alpha = 0.6,
    #    legend = false)
    #savefig("figures/timeseries_subjects_FDG/timeseries_$(i).png")  
end
