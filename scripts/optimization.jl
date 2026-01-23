using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics, Dates
using BlackBoxOptim
using StatsBase; corspearman

# settings
const cent_thresh = nothing  # centiloid threshold for amyloid positivity
const DX = nothing
const suffix = "epiRank"
const epicenter_preprocess = true
const null = true
const zscore_FDG = false

# blackbox settings
maxtime = 60  # seconds of maximum runtime
tracemode = :compact  # :compact, :verbose, :silent. Output of optimization process

# read adjacency matrix from CSV
W = Matrix(CSV.read("data/Schaefer2018_200Parcels_CN.csv", DataFrame; header=false))
N = size(W,1)

# read subjcet ID of people with epicenters
epicenter_IDs, epicenters = read_epicenter_subjects_and_intersections("figures/tau_PET_distributions/subjects_with_epicenters_ALL_metrics.txt")

# load data
println("\nloading dataset...")
FDG_matrix, amyloid_matrix, tau_matrix = nothing, nothing, nothing
FDG_matrix, amyloid_matrix, tau_matrix, subject_IDs = load_dataset(:FDG_amyloid_tau_longitudinal; centiloid_threshold=cent_thresh, DX=DX)

# drop missing rows and normalize
FDG_matrix, amyloid_matrix, tau_matrix, nonmissing_subj = drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)
if zscore_FDG
    FDG_ref = FDG_matrix_reference()  # all FDGs of "CN" subjects as reference for z-scoring
    FDG_matrix = zscore_PET(FDG_matrix, FDG_ref)
else
    FDG_matrix = normalize_rows(FDG_matrix)
end
amyloid_matrix = normalize_rows(amyloid_matrix)
subject_IDs = subject_IDs[nonmissing_subj]  # keep track of subject IDs after dropping missing

# only use individuals with a detected epicenter(s)
if epicenter_preprocess
    epi_idxs = findall(x -> x in epicenter_IDs, subject_IDs)
    FDG_matrix = FDG_matrix[epi_idxs, :]
    amyloid_matrix = amyloid_matrix[epi_idxs, :]
    tau_matrix = tau_matrix[epi_idxs, :]
    subject_IDs = subject_IDs[epi_idxs]
end
println("...using $(size(FDG_matrix,1)) subjects.\n")

# Laplacian
L = laplacian(W, kind=:out, normalize=true)

# Optimization & simulation settings
T = 1.0  # total time for global optimization
tspan = (0, 50)  # time span for timesweep optimization
Tn = 500  # number of timepoints to simulate

# DEFINE METRICS (if necessary)
#m = 10  # number of epicenter candidates
#epicenter_accuracy_m(pred, tau) = epicenter_accuracy(pred, tau, m)  # define epicenter metric if used

# define epicenter accuracy tailored each individual
#3if epicenter_preprocess
#3    epicenter_accuracies = Array{Function}(undef, length(subject_IDs))
#3    for i in eachindex(subject_IDs)
#3        m = length(epicenters[i])  # number of epicenter candidates for this subject
#3        epicenter_accuracies[i] = epicenter_accuracy_m(pred, tau) = epicenter_accuracy(pred, tau, m)  # define epicenter metric if used
#3    end
#3end
if epicenter_preprocess
    rank_means = Array{Function}(undef, length(subject_IDs))
    for i in eachindex(subject_IDs)
        epi_idxs = epicenters[i]  # epicenter indices (experimental)
        rank_means[i] = (pred, tau) -> mean_epi_rank(pred, tau, epi_idxs)
    end
end


# PICK A METRIC, must take two vectors and return a scalar (see epicenter_accuracy_m above)
#metric = epicenter_accuracy_m
#metric = epicenter_accuracies
metric = rank_means
#metric = cor
#metric = corspearman
#metric = top1_rank_score


# empiric PET models
amy_pet_model = [metric[i](-amyloid_matrix[i,:], tau_matrix[i,:]) for i in axes(amyloid_matrix,1)]
FDG_pet_model = [metric[i](-FDG_matrix[i,:], tau_matrix[i,:]) for i in axes(FDG_matrix,1)]
median(amy_pet_model)
median(FDG_pet_model)
# nothing - 15
# 30 - 195
# 28 - 73
# 26 - 125
# 24 - 97.5

# ================================
# RUN FIRST OPTIMIZATION (global eA, eF, individual rho)
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
summary_file = "results/opt_timesweep__centThresh$(cent_thresh)_$(suffix).txt"
save_optimization_summary(summary_file; res=res_timesweep, best_params=best_params_timesweep, best_score=best_score_timesweep, tspan, Tn)
println("Saved optimization results to $summary_file")


# run null if asked
# --- helper (define once, outside the loop) ---
function shuffle_rows_same!(A::AbstractMatrix, B::AbstractMatrix, rng::AbstractRNG)
    @assert size(A) == size(B)
    for i in axes(A, 1)
        p = randperm(rng, size(A, 2))
        A[i, :] .= A[i, p]
        B[i, :] .= B[i, p]
    end
end

# run null if asked
if null
    mkpath("results")

    n_null = 3
    null_scores = Vector{Float64}(undef, n_null)

    # optional: reproducible nulls
    rng = MersenneTwister(1234)

    # keep originals intact
    amy0 = copy(amyloid_matrix)
    fdg0 = copy(FDG_matrix)

    for it in 1:n_null
        # fresh shuffled copies each iteration
        amy = copy(amy0); fdg = copy(fdg0)
        shuffle_rows_same!(amy, fdg, rng)

        objective_timesweep = make_objective_timesweep(
            metric, L, amy, fdg, tau_matrix, tspan, Tn
        )

        res_timesweep = bboptimize(
            objective_timesweep;
            SearchRange = [
                (0.0, 1.0),   # ϵA
                (0.0, 1.0),   # ϵF
            ],
            NumDimensions = 2,
            MaxTime = maxtime,
            TraceMode = tracemode
        )

        best_score_timesweep = -best_fitness(res_timesweep)
        null_scores[it] = best_score_timesweep

        println("Null run $it/$n_null: best objective = $(best_score_timesweep)")
    end

    # save just the best objectives
    null_file = "results/null_opt_objectives_timesweep__centThresh$(cent_thresh)_$(suffix).txt"
    μ   = mean(null_scores)
    mn  = minimum(null_scores)
    mx  = maximum(null_scores)
    open(null_file, "w") do io
        println(io, "# Null optimization best objectives")
        println(io, "# centThresh=$(cent_thresh)  suffix=$(suffix)  n_null=$(n_null)")
        println(io, "# summary statistics")
        println(io, "# mean = $μ")
        println(io, "# min  = $mn")
        println(io, "# max  = $mx")
        println(io)
        println(io, "# one value per line: best_score_timesweep")
        for s in null_scores
            println(io, s)
        end
    end
    println("Saved null best objectives to $null_file")
    println("Null summary: mean=$μ, min=$mn, max=$mx")
end


#
## ================================
## RUN THIRD OPTIMIZATION (individual eA, eF, and rho)
## ================================
#S = size(amyloid_matrix, 1)  # number of subjects
#subject_fits = Vector{NamedTuple}(undef, S)
#
#for i in 1:S
#    println("\nOptimizing subject $i / $S ...")
#
#
#    # DEFINE METRICS for each subject based on how many epicenters they have, this might be good, but will have to check for overfitting which is likely
#    #mi = length(epicenters[i])  # number of epicenter candidates for this subject
#    #epicenter_accuracy_m(pred, tau) = epicenter_accuracy(pred, tau, mi)  # define epicenter metric if used
#
#    # reshape rows to 1×N matrices
#    amy_i = reshape(amyloid_matrix[i, :], 1, :)
#    fdg_i = reshape(FDG_matrix[i, :],     1, :)
#    #amy_i = reshape(rand(N), 1, :)
#    #fdg_i = reshape(rand(N),     1, :)
#    tau_i = reshape(tau_matrix[i, :],     1, :)
#
#    # reuse global objective constructor!
#    objective_i = make_objective_timesweep(
#        metric, L, amy_i, fdg_i, tau_i, tspan, Tn
#    )
#
#    # run BBOpt for this subject
#    res_i = bboptimize(
#        objective_i;
#        SearchRange = [
#            (-1.0, 1.0),   # ϵA
#            (-1.0, 1.0),   # ϵF
#        ],
#        NumDimensions = 2,
#        MaxTime = 10,
#        TraceMode = :silent
#    )
#
#    best_params_i = best_candidate(res_i)
#    best_score_i  = -best_fitness(res_i)
#    
#    # compare with empiric models
#    amy_score_i = metric(vec(amy_i), vec(tau_i))
#    FDG_score_i = metric(vec(fdg_i), vec(tau_i))
#
#    # store subject-level results
#    subject_fits[i] = (
#        ID = subject_IDs[i],
#        best_params = best_params_i,
#        best_score  = best_score_i,
#        res         = res_i,
#        amyloid_score = amy_score_i,
#        FDG_score = FDG_score_i
#    )
#    println("best score: $best_score_i\namyloid PET: $(amy_score_i)\nFDG PET: $(FDG_score_i)\nbest params: $best_params_i\n--------------------\n")
#end
#
##-------------------------------
## Here's a future idea. Take a look at individuals who have a particularly high tau load in a few regions only.
## i.e. from the tau data, find individuals who basically only have tau load in one place. Then we check whether our model can predict that high-tau region.
## this is maybe a better way to test if we can predict epicenters.
## -------------------------------
#file_name = "results/opt_$(string(metric))_individualized_centThresh$(cent_thresh)_$(suffix).csv" 
#df_local, csv_path = save_local_timesweep_results(subject_fits, file_name)
##
#
## ================================
## SAVE EMPIRICAL BASELINE SUMMARY
## ================================
empirical_file = "results/empirical_centThresh$(cent_thresh)_$(suffix).txt"

open(empirical_file, "w") do io
    println(io, "Empirical baseline summary")
    println(io, "Metric: $(string(metric))")
    println(io)
    println(io, "Number of subjects: $(length(amy_pet_model))")
    println(io)
    println(io, "Amyloid PET vs Tau:")
    println(io, "  mean   = $(mean(amy_pet_model))")
    println(io, "  median = $(median(amy_pet_model))")
    println(io, "  std    = $(std(amy_pet_model))")
    println(io)
    println(io, "FDG PET vs Tau:")
    println(io, "  mean   = $(mean(FDG_pet_model))")
    println(io, "  median = $(median(FDG_pet_model))")
    println(io, "  std    = $(std(FDG_pet_model))")
    println(io)
    println(io, "Per-subject values (same order as subject_IDs):")
    println(io)
    println(io, "ID, amyloid_score, FDG_score")
    for i in eachindex(subject_IDs)
        println(io, "$(subject_IDs[i]), $(amy_pet_model[i]), $(FDG_pet_model[i])")
    end
end

println("Saved empirical baseline summary to $empirical_file")

# ==============================================================================
# GLOBAL OPTIMIZATION (global eA, eF, rho)
# ==============================================================================
#objective = make_objective_global(metric, L, amyloid_matrix, FDG_matrix,
#                                  tau_matrix, T)
#res = bboptimize(
#    objective;
#    SearchRange = [
#        (0.0, 50.0),   # ρ
#        (-1.0, 1.0),   # ϵA
#        (-1.0, 1.0),   # ϵF
#    ],
#    NumDimensions = 3,
#    MaxTime = maxtime,
#    TraceMode = tracemode
#)
#best_params = best_candidate(res)
#best_score  = -best_fitness(res)
## save optimization
#summary_file = "results/opt_$(string(metric))_global_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).txt"
#save_optimization_summary(summary_file; res, best_params, best_score, tspan, Tn)
#println("Saved optimization results to $summary_file")