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
tspan = (0,20.0)
Tn = 100

# make predictions
#init_timeseries = disease_initiation_timeseries(L, diagm(vec(mean(amyloid_matrix;dims=1))), diagm(vec(mean(FDG_matrix;dims=1))), ones(N), 1, 1, 1, 0, 0, tspan, Tn)
for i in axes(amyloid_matrix,1)
    amyloid_M = diagm(amyloid_matrix[i,:])
    FDG_M = diagm(FDG_matrix[i,:])
    init_timeseries = disease_initiation_timeseries(L, amyloid_M, FDG_M, ones(N), 1, 0, 1, 0, 0, tspan, Tn)

    # plot timeseries
    t = range(tspan[1], tspan[2], length=Tn)
    plot(t, init_timeseries',  # transpose so each trajectory is a series
        xlabel = "Time",
        ylabel = "u(t)",
        title = "All regions",
        lw = 2,
        alpha = 0.6,
        legend = false)
    savefig("figures/timeseries_subjects_FDG/timeseries_$(i).png")  
end
