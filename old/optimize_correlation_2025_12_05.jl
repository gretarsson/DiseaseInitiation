#######################################################################
# OPTIMIZE MODEL PARAMETERS FOR MAXIMUM CORRELATION WITH TAU
#######################################################################

using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics, Dates
using BlackBoxOptim
maxtime = 600

# ---------------------------------------------------------
# LOAD STRUCTURAL CONNECTOME
# ---------------------------------------------------------
W = Matrix(CSV.read("data/Schaefer2018_200Parcels_CN.csv", DataFrame; header=false))
N = size(W, 1)

# ---------------------------------------------------------
# LOAD PET DATA
# ---------------------------------------------------------
FDG_matrix, amyloid_matrix, tau_matrix = load_dataset(:FDG_amyloid_tau_longitudinal)
FDG_matrix, amyloid_matrix, tau_matrix = drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)

# ---------------------------------------------------------
# NORMALIZE ROWS OF PET MATRICES TO [0,1]
# ---------------------------------------------------------
function normalize_rows!(M)
    for i in axes(M,1)
        row = M[i, :]
        M[i, :] = (row .- minimum(row)) ./ (maximum(row) - minimum(row))
    end
end

normalize_rows!(FDG_matrix)
normalize_rows!(amyloid_matrix)

S = size(tau_matrix, 1)

# ---------------------------------------------------------
# BUILD LAPLACIAN
# ---------------------------------------------------------
L = laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))  # normalize spectral radius

# ---------------------------------------------------------
# SIMULATION SETTINGS
# ---------------------------------------------------------
tspan = (0, 50.0)
Tn = 500

#######################################################################
# OBJECTIVE 1: TIME SERIES — maximize correlation with tau
#######################################################################

"""
Compute the maximum correlation over time between model prediction
and empirical tau for a single subject.
"""
function best_correlation_over_time(init_timeseries, tau_vec)
    N, T = size(init_timeseries)
    corrs = zeros(T)

    for t in 1:T
        pred = init_timeseries[:, t]
        if std(pred) == 0
            corrs[t] = 0
        else
            corrs[t] = cor(pred, tau_vec)

            # For Spearman:
            # corrs[t] = corspearman(pred, tau_vec)
        end
    end

    return maximum(corrs)
end


"""
Objective for time-series optimization:
Given θ = (ϵA, ϵF), simulate each subject and compute the best correlation.
"""
function objective_timeseries(θ)
    ϵA, ϵF = θ
    corrs = zeros(S)

    for i in 1:S
        init_timeseries = disease_initiation_timeseries(
            L,
            diagm(amyloid_matrix[i,:]),
            diagm(FDG_matrix[i,:]),
            ones(N),
            1, ϵA, ϵF, 0, 0,
            tspan,
            Tn
        )

        corrs[i] = best_correlation_over_time(init_timeseries, tau_matrix[i,:])
    end

    return -mean(corrs)   # minimize negative correlation
end


println("\n===============================")
println("Running time-series optimization")
println("===============================\n")

res_ts = bboptimize(
    objective_timeseries;
    SearchRange=[(-1.0,5.0), (-1.0,5.0)],
    NumDimensions=2,
    MaxTime=maxtime,
    TraceMode=:verbose
)

best_params_ts = best_candidate(res_ts)
best_score_ts = -best_fitness(res_ts)

println("\nBest parameters (time series):")
println(best_params_ts)
println("Best mean correlation = ", best_score_ts)

# ---------------------------------------------------------
# NULL MODELS
# ---------------------------------------------------------
null_corr_amy = [cor(amyloid_matrix[i,:], tau_matrix[i,:]) for i in 1:S]
null_corr_fdg = [cor(FDG_matrix[i,:], tau_matrix[i,:]) for i in 1:S]

# ---------------------------------------------------------
# SAVE SUMMARY FILE
# ---------------------------------------------------------
summary_file_ts = "results/correlation_timeseries_summary_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).txt"

open(summary_file_ts, "w") do io
    println(io, "Time-series correlation optimization")
    println(io, "Timestamp: ", now())
    println(io, "Best ϵA = ", best_params_ts[1])
    println(io, "Best ϵF = ", best_params_ts[2])
    println(io, "Best mean correlation = ", best_score_ts)
    println(io, "\nNull correlations:")
    println(io, "Amyloid PET mean correlation = ", mean(null_corr_amy))
    println(io, "FDG PET mean correlation = ", mean(null_corr_fdg))
    println(io, "\nBlackBoxOptim summary:\n")
    println(io, res_ts)
end


# ---------------------------------------------------------
# RECONSTRUCT OPTIMAL MODEL RESULTS FOR PLOTTING
# ---------------------------------------------------------
optimal_corrs = zeros(S)
for i in 1:S
    init_timeseries = disease_initiation_timeseries(
        L,
        diagm(amyloid_matrix[i,:]),
        diagm(FDG_matrix[i,:]),
        ones(N),
        1, best_params_ts[1], best_params_ts[2], 0, 0,
        tspan,
        Tn
    )
    optimal_corrs[i] = best_correlation_over_time(init_timeseries, tau_matrix[i,:])
end

# ---------------------------------------------------------
# HISTOGRAMS
# ---------------------------------------------------------
histogram(null_corr_fdg, alpha=0.5, label="FDG null (mean = $(round(mean(null_corr_fdg), digits=2)))")
histogram!(null_corr_amy, alpha=0.5, label="Amyloid null (mean = $(round(mean(null_corr_amy), digits=2)))")
histogram!(
    optimal_corrs,
    alpha=0.6,
    label="Optimized model (mean = $(round(mean(optimal_corrs), digits=2)))",
    title="Correlation with Tau",
    xlabel="Correlation",
    ylabel="Frequency"
)

savefig("figures/correlation_timeseries_histogram_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).png")


#######################################################################
# OPTIONAL SECOND-STAGE OPTIMIZATION:
# A SINGLE-TIMEPOINT MODEL WITH GLOBAL ρ
#######################################################################
# (Kept for completeness; comment out if not needed)
#######################################################################

println("\n===============================")
println("Running single-timepoint optimization")
println("===============================\n")

function objective_single(θ)
    ρ, ϵA, ϵF = θ
    init_matrix = disease_initiation_matrix(
        L,
        amyloid_matrix,
        FDG_matrix,
        ones(N),
        ρ, ϵA, ϵF, 0, 0,
        1.0    # T
    )
    corrs = zeros(S)
    for i in 1:S
        pred = init_matrix[i,:]
        corrs[i] = std(pred) == 0 ? 0 : cor(pred, tau_matrix[i,:])
    end
    return -mean(corrs)
end

res_single = bboptimize(
    objective_single;
    SearchRange=[(0.0,50.0), (-1.0,5.0), (-1.0,5.0)],
    NumDimensions=3,
    MaxTime=maxtime,
    TraceMode=:verbose
)

best_params_single = best_candidate(res_single)
best_score_single = -best_fitness(res_single)

println("Best single-time parameters: ", best_params_single)
println("Best mean correlation = ", best_score_single)

# Recompute model
init_matrix_best = disease_initiation_matrix(
    L,
    amyloid_matrix,
    FDG_matrix,
    ones(N),
    best_params_single[1],
    best_params_single[2],
    best_params_single[3],
    0, 0,
    1.0
)

cor_single = [cor(init_matrix_best[i,:], tau_matrix[i,:]) for i in 1:S]

# Plot
histogram(null_corr_fdg, alpha=0.5, label="FDG null")
histogram!(null_corr_amy, alpha=0.5, label="Amyloid null")
histogram!(
    cor_single,
    alpha=0.6,
    label="Optimized single-time model (mean = $(round(mean(cor_single), digits=2)))",
    title="Correlation with Tau (Global ρ)",
    xlabel="Correlation",
    ylabel="Frequency"
)
savefig("figures/correlation_single_histogram_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).png")
