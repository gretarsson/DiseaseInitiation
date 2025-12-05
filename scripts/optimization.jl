using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics, Dates
using BlackBoxOptim
using StatsBase; corspearman

# blackbox settings
maxtime = 300  # seconds of maximum runtime
tracemode = :compact  # :compact, :verbose, :silent. Output of optimization process

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
#metric = corspearman
metric = top1_rank_score

# empiric PET models
amy_pet_model = [metric(amyloid_matrix[i,:], tau_matrix[i,:]) for i in axes(amyloid_matrix,1)]
FDG_pet_model = [metric(FDG_matrix[i,:], tau_matrix[i,:]) for i in axes(FDG_matrix,1)]
println("Mean empirical amyloid PET model score: $(mean(amy_pet_model))")
println("Mean empirical FDG PET model score: $(mean(FDG_pet_model))")

# ================================
# RUN FIRST OPTIMIZATION (global eA, eF, local rho)
# ================================
objective_timesweep = make_objective_timesweep(metric, L, amyloid_matrix, FDG_matrix,
                                     tau_matrix, tspan, Tn)
res_timesweep = bboptimize(
    objective_timesweep;
    SearchRange = [
        (-1.0, 1.0),   # ϵA
        (-1.0, 1.0),   # ϵF
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
# GLOBAL OPTIMIZATION (global eA, eF, rho)
# ==============================================================================
objective = make_objective_global(metric, L, amyloid_matrix, FDG_matrix,
                                  tau_matrix, T)
res = bboptimize(
    objective;
    SearchRange = [
        (0.0, 50.0),   # ρ
        (-1.0, 1.0),   # ϵA
        (-1.0, 1.0),   # ϵF
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

# ================================
# RUN THIRD OPTIMIZATION (local eA, eF, rho)
# ================================
S = size(amyloid_matrix, 1)  # number of subjects
subject_fits = Vector{NamedTuple}(undef, S)

for i in 1:S
    println("\nOptimizing subject $i / $S ...")

    # reshape rows to 1×N matrices
    amy_i = reshape(amyloid_matrix[i, :], 1, :)
    fdg_i = reshape(FDG_matrix[i, :],     1, :)
    #amy_i = reshape(rand(N), 1, :)
    #fdg_i = reshape(rand(N),     1, :)
    tau_i = reshape(tau_matrix[i, :],     1, :)

    # reuse global objective constructor!
    objective_i = make_objective_timesweep(
        metric, L, amy_i, fdg_i, tau_i, tspan, Tn
    )

    # run BBOpt for this subject
    res_i = bboptimize(
        objective_i;
        SearchRange = [
            (-1.0, 1.0),   # ϵA
            (-1.0, 1.0),   # ϵF
        ],
        NumDimensions = 2,
        MaxTime = 10,
        TraceMode = :silent
    )

    best_params_i = best_candidate(res_i)
    best_score_i  = -best_fitness(res_i)
    
    # compare with empiric models
    amy_score_i = metric(vec(amy_i), vec(tau_i))
    FDG_score_i = metric(vec(fdg_i), vec(tau_i))

    # store subject-level results
    subject_fits[i] = (
        ID = subject_IDs[i],
        best_params = best_params_i,
        best_score  = best_score_i,
        res         = res_i,
        amyloid_score = amy_score_i,
        FDG_score = FDG_score_i
    )
    println("best score: $best_score_i\namyloid PET: $(amy_score_i)\nFDG PET: $(FDG_score_i)\nbest params: $best_params_i\n--------------------\n")
end

#-------------------------------
# Here's a future idea. Take a look at individuals who have a particularly high tau load in a few regions only.
# i.e. from the tau data, find individuals who basically only have tau load in one place. Then we check whether our model can predict that high-tau region.
# this is maybe a better way to test if we can predict epicenters.
# -------------------------------
file_name = "results/opt_$(string(metric))_local_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).csv" 
df_local, csv_path = save_local_timesweep_results(subject_fits, file_name)
