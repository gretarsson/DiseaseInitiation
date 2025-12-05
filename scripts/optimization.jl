using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics, Dates
using BlackBoxOptim
using StatsBase; corspearman

# blackbox settings
maxtime = 3  # seconds of maximum runtime
tracemode = :compact  # :compact, :verbose, :silent. Output of optimization process

# read adjacency matrix from CSV
W = Matrix(CSV.read("data/Schaefer2018_200Parcels_CN.csv", DataFrame; header=false))
N = size(W,1)

# load data
println("\nloading dataset...")
FDG_matrix, amyloid_matrix, tau_matrix = nothing, nothing, nothing
FDG_matrix, amyloid_matrix, tau_matrix = load_dataset(:FDG_amyloid_tau_longitudinal; centiloid_threshold=20)
println("...using $(size(FDG_matrix,1)) subjects.\n")

# drop missing rows and normalize
FDG_matrix, amyloid_matrix, tau_matrix = drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)
FDG_matrix = normalize_rows(FDG_matrix)
amyloid_matrix = normalize_rows(amyloid_matrix)

# Laplacian
L = laplacian(W, kind=:out, normalize=true)

# Optimization & simulation settings
T = 1.0  # total time for global optimization
tspan = (0, 50)  # time span for timesweep optimization
Tn = 500  # number of timepoints to simulate

# DEFINE METRICS (if necessary)
m = 10  # number of epicenter candidates
epicenter_accuracy_m(pred, tau) = epicenter_accuracy(pred, tau, m)  # define epicenter metric if used

# PICK A METRIC
#metric = epicenter_accuracy_m
#metric = cor
metric = corspearman

# empiric PET models
amy_pet_model = [metric(amyloid_matrix[i,:], tau_matrix[i,:]) for i in axes(amyloid_matrix,1)]
FDG_pet_model = [metric(FDG_matrix[i,:], tau_matrix[i,:]) for i in axes(FDG_matrix,1)]

# ================================
# RUN FIRST OPTIMIZATION (same parameters)
# ================================
objective_timesweep = make_objective_timesweep(metric, L, amyloid_matrix, FDG_matrix,
                                     tau_matrix, tspan, Tn)
res_timesweep = bboptimize(
    objective_timesweep;
    SearchRange = [
        (-1.0, 5.0),   # ϵA
        (-1.0, 5.0),   # ϵF
    ],
    NumDimensions = 2,
    MaxTime = maxtime,
    TraceMode = tracemode
)
best_params_timesweep = best_candidate(res_timesweep)
best_score_timesweep  = -best_fitness(res_timesweep)
# save optimization
summary_file = "results/opt_$(string(metric))_timesweep_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).txt"
save_optimization_summary(summary_file; res=res_timesweep, best_params=best_params_timesweep, best_score=best_score_timesweep, tspan, Tn)
println("Saved optimization results to $summary_file")

# ==============================================================================
# GLOBAL OPTIMIZATION
# ==============================================================================
objective = make_objective_global(metric, L, amyloid_matrix, FDG_matrix,
                                  tau_matrix, T)
res = bboptimize(
    objective;
    SearchRange = [
        (0.0, 50.0),   # ρ
        (-1.0, 5.0),   # ϵA
        (-1.0, 5.0),   # ϵF
    ],
    NumDimensions = 3,
    MaxTime = maxtime,
    TraceMode = tracemode
)
best_params = best_candidate(res)
best_score  = -best_fitness(res)
# save optimization
summary_file = "results/opt_$(string(metric))_global_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).txt"
save_optimization_summary(summary_file; res, best_params, best_score, tspan, Tn)
println("Saved optimization results to $summary_file")
