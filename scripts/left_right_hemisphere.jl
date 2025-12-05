using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics, Dates
using BlackBoxOptim
default(dpi=300)  # set default dpi for plots


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


# Generate Laplacian and activity matrix
L = laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))  # normalize Laplacian
T = 1.0
tspan = (0, 50.0)
Tn = 500
skip = 10
m = 10  # number of epicenter candidates
S = size(tau_matrix,1)

# divide into left and right hemisphere alzheimers
function LR_ratio(M)
    LR_ratio = zeros(size(M,1))
    for i in axes(M,1)
        RH = sum(M[i,1:100])
        LH = sum(M[i,101:200])
        LR_ratio[i] = (RH - LH) / (RH + LH)
    end
    return LR_ratio
end
# Investigate the empirical LR ratio distribution in tau PET
LR_ratio_emp = LR_ratio(tau_matrix)
qlow, qhigh = quantile(LR_ratio_emp, [0.20,0.80])
histogram(LR_ratio_emp;
    bins = 30,
    xlabel = "LR ratio",
    ylabel = "Count",
    title = "Distribution of LR Ratio",
    alpha = 0.6,
    legend = false
);
vline!([qlow], color=:black, lw=2);
vline!([qhigh], color=:black, lw=2);
savefig("figures/LR_ratio_distribution_tauPET.png")

# OPTIMIZE LEFT-RIGHT HEMISPHERE RATIO
function objective(θ)
    ϵA, ϵF = θ
    optimal_ratios = zeros(S)  # store the optimal epicenter ratio per subject over time, given parameters
    for i in 1:S
        # build predictions
        init_timeseries = disease_initiation_timeseries(
            L,
            diagm(amyloid_matrix[i,:]),
            diagm(FDG_matrix[i,:]),
            1/1*ones(N),
            1, ϵA, ϵF, 0, 0,
            tspan,
            Tn
        )
        LR_ratios_i = LR_ratio(transpose(init_timeseries[:,skip:end]))  
        # find best time point
        optimal_ratio_ind = argmin(abs.(LR_ratios_i .- LR_ratio_emp[i]))
        optimal_ratios[i] = LR_ratios_i[optimal_ratio_ind]
        # average the time points
        #optimal_ratios[i] = mean(LR_ratios_i)
    end
    spearman_r, _ = spearman_with_pvalue(LR_ratio_emp, optimal_ratios)
    return -spearman_r           # minimize → negative to maximize accuracy
end

res = bboptimize(
    objective;
    SearchRange = [
        (-1.0, 5),   # ϵA
        (-1.0, 5),   # ϵF
    ],
    NumDimensions = 2,
    MaxTime = 600,      # maximal run time (seconds)
    TraceMode = :verbose
)

best_params = best_candidate(res)
best_score  = -best_fitness(res)

# NULL MODEL
# copmare how empirical PET does in comparison
LR_ratio_null_FDG = LR_ratio(FDG_matrix)  # null model based on FDG
LR_ratio_null_amy = LR_ratio(amyloid_matrix)  # null model based on amyloid
ρ_fdg_null, p_fdg_null = spearman_with_pvalue(LR_ratio_emp, LR_ratio_null_FDG)
ρ_amy_null, p_amy_null = spearman_with_pvalue(LR_ratio_emp, LR_ratio_null_amy)



# Write a summary statistics file
# ============================
# WRITE SUMMARY STATISTICS FILE
# ============================

mkpath("results")
summary_file = "results/optimal_LRratio_summary_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).txt"

open(summary_file, "w") do io
    println(io, "========================================")
    println(io, "  OPTIMIZATION SUMMARY: LR RATIO MODEL  ")
    println(io, "========================================")
    println(io, "Timestamp: ", Dates.now())
    println(io)

    # Best parameters
    println(io, "Best parameters:")
    println(io, "  ϵA = $(best_params[1])    # amyloid scaling")
    println(io, "  ϵF = $(best_params[2])    # FDG scaling")
    println(io)

    # Best achieved score
    println(io, "Best score (Spearman ρ): ", best_score)
    println(io)

    # Null model results
    println(io, "Null model performance:")
    println(io, "  Amyloid PET LR ratio:")
    println(io, "    ρ = $(ρ_amy_null)")
    println(io, "    p = $(p_amy_null)")
    println(io)
    println(io, "  FDG PET LR ratio:")
    println(io, "    ρ = $(ρ_fdg_null)")
    println(io, "    p = $(p_fdg_null)")
    println(io)

    # BlackBoxOptim summary
    println(io, "----------------------------------------")
    println(io, "BlackBoxOptim summary:")
    println(io, res)
    println(io, "----------------------------------------")
end

println("Saved optimization results to $summary_file")


# These results corroborate the following papers investigating hemispheric asymmetry in Alzheimer's disease:
# https://www.nature.com/articles/s41467-025-63564-2
# amyloid positive correlation with tau
# https://pubmed.ncbi.nlm.nih.gov/40879412/
# FDG negative correlation with tau

# check the optimal ratios with the best parameters
optimal_ratios = zeros(S)  # store the optimal epicenter ratio per subject over time, given parameters
for i in 1:S
    # build predictions
    init_timeseries = disease_initiation_timeseries(
        L,
        diagm(amyloid_matrix[i,:]),
        diagm(FDG_matrix[i,:]),
        1/1*ones(N),
        1, best_params[1], best_params[2], 0, 0,
        #1, -0.91, 1.1, 0, 0,
        tspan,
        Tn
    )
    LR_ratios_i = LR_ratio(transpose(init_timeseries[:,skip:end]))
    optimal_ratio_ind = argmin(abs.(LR_ratios_i .- LR_ratio_emp[i]))
    optimal_ratios[i] = LR_ratios_i[optimal_ratio_ind]
end
spearman_r, _ = spearman_with_pvalue(LR_ratio_emp, optimal_ratios)
p = scatter(optimal_ratios, LR_ratio_emp;
    xlabel = "Optimal LR Ratio (Model)",
    ylabel = "Empirical LR Ratio (Tau PET)",
    title = "Optimal vs Empirical LR Ratios",
    label = "Spearman ρ = $(round(spearman_r, digits=2))",
    dpi=300
)
savefig("figures/Optimal_vs_Empirical_LR_Ratios_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).png")


# ---------------------------------------------------------------------------------------------------------
# GLOBAL OPTIMIZATION
# -------------------------------------------------------------------------------------------------
function objective(θ)
    ρ, ϵA, ϵF = θ
    init_matrix = disease_initiation_matrix(L, 
        amyloid_matrix,
        FDG_matrix,
        1/1*ones(N),
        ρ, ϵA, ϵF, 0, 0,
        T
    )
    LR_ratio_model = LR_ratio(init_matrix)
    spearman_r, _ = spearman_with_pvalue(LR_ratio_emp, LR_ratio_model)
    return -spearman_r           # minimize → negative to maximize accuracy
end

res = bboptimize(
    objective;
    SearchRange = [
        (0, 50),     # ρ
        (-1.0, 20),   # ϵA
        (-1.0, 20),   # ϵF
    ],
    NumDimensions = 3,
    MaxTime = 600,      # maximal run time (seconds)
    TraceMode = :verbose
)

best_params = best_candidate(res)
best_score  = -best_fitness(res)


# Write a summary statistics file
# ============================
# WRITE SUMMARY STATISTICS FILE
# ============================

mkpath("results")
summary_file = "results/optimal_LRratio_global_summary_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).txt"

open(summary_file, "w") do io
    println(io, "========================================")
    println(io, "  OPTIMIZATION SUMMARY: LR RATIO MODEL  ")
    println(io, "========================================")
    println(io, "Timestamp: ", Dates.now())
    println(io)

    # Best parameters
    println(io, "Best parameters:")
    println(io, "  ϵA = $(best_params[1])    # amyloid scaling")
    println(io, "  ϵF = $(best_params[2])    # FDG scaling")
    println(io)

    # Best achieved score
    println(io, "Best score (Spearman ρ): ", best_score)
    println(io)

    # BlackBoxOptim summary
    println(io, "----------------------------------------")
    println(io, "BlackBoxOptim summary:")
    println(io, res)
    println(io, "----------------------------------------")
end

println("Saved optimization results to $summary_file")
# PLOT BEST RESULT
# check the optimal ratios with the best parameters
optimal_ratios = zeros(S)  # store the optimal epicenter ratio per subject over time, given parameters
init_matrix = disease_initiation_matrix(L, 
    amyloid_matrix,
    FDG_matrix,
    1/1*ones(N),
    best_params[1], best_params[2], best_params[3], 0, 0,
    T
)
LR_ratio_model = LR_ratio(init_matrix)
spearman_r, _ = spearman_with_pvalue(LR_ratio_emp, LR_ratio_model)
p = scatter(LR_ratio_model, LR_ratio_emp;
    xlabel = "Optimal LR Ratio (Model, global time)",
    ylabel = "Empirical LR Ratio (Tau PET)",
    title = "Optimal vs Empirical LR Ratios",
    label = "Spearman ρ = $(round(spearman_r, digits=2))",
    dpi=300
)
savefig("figures/Optimal_vs_Empirical_LR_Ratios_global_time_$(Dates.format(now(), "yyyy-mm-dd_HHMM")).png")
