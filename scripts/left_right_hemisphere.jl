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
FDG_matrix, amyloid_matrix, tau_matrix = load_dataset(:FDG_amyloid_tau_longitudinal)

# drop missing rows
FDG_matrix, amyloid_matrix, tau_matrix = drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)
if FDG_matrix !== nothing
    FDG_matrix ./= maximum(FDG_matrix)  # normalize by median
end
if amyloid_matrix !== nothing
    amyloid_matrix ./= maximum(amyloid_matrix)  # normalize by median
end

# Generate Laplacian and activity matrix
L = laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))  # normalize Laplacian
T = 1.0

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
vline!([qhigh], color=:black, lw=2)


# make predictions
ρ = 1.0; ϵ = 0.1; k = 1.0; λ = 1.0
init_matrix_FDG = disease_initiation_matrix(L, FDG_matrix, ones(N), 1, 1, 1, 1, T)
init_matrix_amy = disease_initiation_matrix(L, amyloid_matrix, ones(N), 1, 1, 1, 1, T)

# compute LR ratio for predictions
LR_ratio_FDG = LR_ratio(init_matrix_FDG)
LR_ratio_amy = LR_ratio(init_matrix_amy)


# correlations of LR ratios
using HypothesisTests   # <-- added
# compute correlations and p-values
test_fdg = CorrelationTest(LR_ratio_emp, LR_ratio_FDG)
test_amy = CorrelationTest(LR_ratio_emp, LR_ratio_amy)
test_fdg
r_fdg = test_fdg.r
p_fdg = pvalue(test_fdg)
r_amy = test_amy.r
p_amy = pvalue(test_amy)
# plot correlations
scatter(LR_ratio_emp, LR_ratio_FDG;
    xlabel = "Empirical LR ratio (tau)",
    ylabel = "Predicted LR ratio (FDG-based)",
    title = "r=$(round(r_fdg, digits=3)), p=$(round(p_fdg, digits=3))",
    legend = false
)
savefig("figures/FDG_LR_ratio_scatter.png") 
scatter(LR_ratio_emp, LR_ratio_amy;
    xlabel = "Empirical LR ratio (tau)",
    ylabel = "Predicted LR ratio (amyloid-based)",
    title = "r=$(round(r_amy, digits=3)), p=$(round(p_amy, digits=3))",
    legend = false
)
savefig("figures/AMY_LR_ratio_scatter.png") 


# Check classification accuracy
upper_q = 0.75
lower_q = 0.25
function classify_by_quantiles(v, qlow, qhigh)
    lower, upper = quantile(v, [qlow,qhigh])
    classifications = Vector{Symbol}(undef, length(v))
    for (i,x) in enumerate(v)
        if x < lower
            classifications[i] = :left
        elseif x > upper
            classifications[i] = :right
        else
            classifications[i] = :bilateral
        end
    end
    return classifications
end

emp_classes = classify_by_quantiles(LR_ratio_emp, lower_q, upper_q)
FDG_classes = classify_by_quantiles(LR_ratio_FDG, lower_q, upper_q)
amy_classes = classify_by_quantiles(LR_ratio_amy, lower_q, upper_q)
class_accuracy_FDG = mean(emp_classes .== FDG_classes)
class_accuracy_amy = mean(emp_classes .== amy_classes)


# SPREARMAN
using StatsBase, Distributions

function spearman_with_pvalue(x, y)
    ρ = corspearman(x, y)
    n = length(x)
    # t-statistic
    t = ρ * sqrt((n - 2) / (1 - ρ^2))
    # two-sided p-value
    p = 2 * (1 - cdf(TDist(n - 2), abs(t)))
    return ρ, p
end

ρ_fdg, p_fdg = spearman_with_pvalue(LR_ratio_emp, LR_ratio_FDG)
ρ_amy, p_amy = spearman_with_pvalue(LR_ratio_emp, LR_ratio_amy)



# copmare how empirical PET does in comparison
LR_ratio_null_FDG = LR_ratio(FDG_matrix)  # null model based on FDG
LR_ratio_null_amy = LR_ratio(amyloid_matrix)  # null model based on amyloid
ρ_fdg_null, p_fdg_null = spearman_with_pvalue(LR_ratio_emp, LR_ratio_null_FDG)
ρ_amy_null, p_amy_null = spearman_with_pvalue(LR_ratio_emp, LR_ratio_null_amy)


# plot PET correlations
# compute correlations and p-values
test_fdg = CorrelationTest(LR_ratio_emp, LR_ratio_null_FDG)
test_amy = CorrelationTest(LR_ratio_emp, LR_ratio_null_amy)
test_fdg
r_fdg = test_fdg.r
p_fdg = pvalue(test_fdg)
r_amy = test_amy.r
p_amy = pvalue(test_amy)
scatter(LR_ratio_emp, LR_ratio_null_FDG;
    xlabel = "Empirical LR ratio (tau)",
    ylabel = "Empirical LR ratio (FDG)",
    title = "r=$(round(r_fdg, digits=3)), p=$(round(p_fdg, digits=3))",
    legend = false
)
savefig("figures/PET_FDG_LR_ratio_scatter.png") 
scatter(LR_ratio_emp, LR_ratio_null_amy;
    xlabel = "Empirical LR ratio (tau)",
    ylabel = "Prediempirical LR ratio (amyloid)",
    title = "r=$(round(r_amy, digits=3)), p=$(round(p_amy, digits=3))",
    legend = false
)
savefig("figures/PET_AMY_LR_ratio_scatter.png") 

# we can get to similar results by considering only the empirical, the dynamical systems doesn't give us anything more

# Write a summary statistics file
# ============================
# WRITE SUMMARY STATISTICS FILE
# ============================

mkpath("results")

summary_path = "results/LEFT-RIGHT_summary_statistics.txt"
open(summary_path, "w") do io
    println(io, "SUMMARY STATISTICS")
    println(io, "Generated: $(Dates.now())")
    println(io, "----------------------------------------")

    # Sample size
    println(io, "Number of subjects: $(length(LR_ratio_emp))")
    println(io)

    # Quantile boundaries (empirical)
    println(io, "Empirical LR quantiles:")
    println(io, "  20th percentile: $qlow")
    println(io, "  80th percentile: $qhigh")
    println(io)

    # Pearson correlations
    println(io, "Pearson correlations:")
    println(io, "  FDG-based model vs tau:     r = $(round(r_fdg, digits=4)), p = $(round(p_fdg, digits=4))")
    println(io, "  Amyloid-based model vs tau: r = $(round(r_amy, digits=4)), p = $(round(p_amy, digits=4))")
    println(io)

    # Spearman correlations
    println(io, "Spearman correlations:")
    println(io, "  FDG-based model vs tau:     ρ = $(round(ρ_fdg, digits=4)), p = $(round(p_fdg, digits=4))")
    println(io, "  Amyloid-based model vs tau: ρ = $(round(ρ_amy, digits=4)), p = $(round(p_amy, digits=4))")
    println(io)

    # Null-model Spearman correlations
    println(io, "Null-model Spearman correlations:")
    println(io, "  FDG PET vs tau:     ρ = $(round(ρ_fdg_null, digits=4)), p = $(round(p_fdg_null, digits=4))")
    println(io, "  Amyloid PET vs tau: ρ = $(round(ρ_amy_null, digits=4)), p = $(round(p_amy_null, digits=4))")
    println(io)

    println(io)

    # Classification accuracy
    println(io, "Quantile-based classification accuracy (tercile agreement):")
    println(io, "  FDG prediction accuracy:     $(round(class_accuracy_FDG, digits=4))")
    println(io, "  Amyloid prediction accuracy: $(round(class_accuracy_amy, digits=4))")
    println(io, "  Chance baseline (25/50/25):  0.375")
    println(io)

    # Basic summary of LR distributions
    println(io, "LR ratio summary statistics:")
    println(io, "  Empirical tau:   mean=$(mean(LR_ratio_emp)), std=$(std(LR_ratio_emp))")
    println(io, "  FDG-predicted:   mean=$(mean(LR_ratio_FDG)), std=$(std(LR_ratio_FDG))")
    println(io, "  Amy-predicted:   mean=$(mean(LR_ratio_amy)), std=$(std(LR_ratio_amy))")
    println(io)

    println(io, "----------------------------------------")
    println(io, "End of summary.")
end

println("Summary statistics saved to $(summary_path)")

# These results corroborate the following papers investigating hemispheric asymmetry in Alzheimer's disease:
# https://www.nature.com/articles/s41467-025-63564-2
# amyloid positive correlation with tau
# https://pubmed.ncbi.nlm.nih.gov/40879412/
# FDG negative correlation with tau