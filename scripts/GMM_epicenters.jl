
using DiseaseInitiation
using CSV, DataFrames
using Statistics, StatsBase
using Distributions
using Plots

const DATA_CSV = "data/ADNI_HABS_amyloid_FDG_longitudinal_tau.csv"

const OUTDIR_SAFE = "figures/tau_pet_distrubutions_ren"
const OUTDIR_EXACT = "figures/tau pet distrubutions_ ren"   

mkpath(OUTDIR_SAFE)
mkpath(OUTDIR_EXACT)

const Z_THR = 1.645          
const MAX_ITERS = 300
const TOL = 1e-7
const EPS = 1e-9


sanitize_id(s) = replace(string(s), r"[^\w\-]+" => "_")

"""
Fit 1-Gaussian MLE for vector x (Float64).
Returns (μ, σ, loglik, bic)
"""
function fit_1gauss(x::Vector{Float64})
    n = length(x)
    μ = mean(x)
    σ = std(x)
    σ = (σ < EPS) ? EPS : σ
    ll = sum(logpdf.(Normal(μ, σ), x))
    p = 2  # μ, σ
    bic = -2ll + p * log(n)
    return μ, σ, ll, bic
end

"""
Fit 2-component univariate Gaussian mixture via EM.
Returns (w, μ1, σ1, μ2, σ2, loglik, bic, converged)
Component order is NOT guaranteed; we reorder later by means.
"""
function fit_2gauss_em(x::Vector{Float64};
                       max_iters::Int = MAX_ITERS,
                       tol::Float64 = TOL)

    n = length(x)
    n < 10 && return (0.5, mean(x), std(x)+EPS, mean(x), std(x)+EPS, -Inf, Inf, false)

    # init using quantiles (robust)
    q25, q75 = quantile(x, (0.25, 0.75))
    μ1, μ2 = q25, q75
    σ0 = std(x)
    σ1 = (σ0 < EPS) ? 1.0 : σ0
    σ2 = σ1
    w = 0.5

    ll_prev = -Inf
    converged = false

    for it in 1:max_iters
        # E-step
        d1 = Normal(μ1, max(σ1, EPS))
        d2 = Normal(μ2, max(σ2, EPS))
        p1 = w .* pdf.(d1, x)
        p2 = (1 - w) .* pdf.(d2, x)
        denom = p1 .+ p2 .+ EPS
        r1 = p1 ./ denom
        r2 = 1 .- r1

        # M-step
        w = clamp(mean(r1), 1e-3, 1 - 1e-3)
        s1 = sum(r1) + EPS
        s2 = sum(r2) + EPS

        μ1 = sum(r1 .* x) / s1
        μ2 = sum(r2 .* x) / s2

        v1 = sum(r1 .* (x .- μ1).^2) / s1
        v2 = sum(r2 .* (x .- μ2).^2) / s2
        σ1 = sqrt(max(v1, EPS))
        σ2 = sqrt(max(v2, EPS))

        # log-likelihood
        ll = sum(log.(w .* pdf.(Normal(μ1, σ1), x) .+ (1 - w) .* pdf.(Normal(μ2, σ2), x) .+ EPS))

        if abs(ll - ll_prev) < tol * (1 + abs(ll_prev))
            converged = true
            ll_prev = ll
            break
        end
        ll_prev = ll
    end

    ll = ll_prev
    p = 5  # w, μ1, σ1, μ2, σ2  (w has 1 DOF)
    bic = -2ll + p * log(n)
    return w, μ1, σ1, μ2, σ2, ll, bic, converged
end

"""
Given τ value x and two Gaussians (nonpath, path), compute paper TPI:
TPI = CDF(path, x) - (1 - CDF(nonpath, x)) = CDF(path,x) + CDF(nonpath,x) - 1
"""
function tpi_value(x::Float64, d_non::Normal, d_path::Normal)
    p_non  = cdf(d_non, x)
    p_path = cdf(d_path, x)
    tpi = p_path + p_non - 1
    return clamp(tpi, -1.0, 1.0)
end

"""
Within-subject zscore (ignores missing); returns Vector{Union{Missing,Float64}}
"""
function zscore_within_subject(v::Vector{Union{Missing,Float64}})
    vals = collect(skipmissing(v))
    if isempty(vals)
        return fill(missing, length(v))
    end
    μ = mean(vals)
    σ = std(vals)
    if σ < EPS
        # all same → z=0
        return [x === missing ? missing : 0.0 for x in v]
    end
    return [x === missing ? missing : (x - μ)/σ for x in v]
end

# -----------------------------
# Load ROI names (tau columns) from CSV to label plots
# -----------------------------
df_head = CSV.read(DATA_CSV, DataFrame; limit=1)
tau_cols = filter(c -> startswith(c, "tau.SUVR.Schaefer200"), names(df_head))
@assert !isempty(tau_cols) "No tau columns found with prefix tau.SUVR.Schaefer200 in $DATA_CSV"
roi_names = String.(tau_cols)

# -----------------------------
# Load matrices using the same dataset key as your mentor
# IMPORTANT: no centiloid threshold (include all)
# -----------------------------
FDG_matrix, amyloid_matrix, tau_matrix, subject_IDs =
    load_dataset(:FDG_amyloid_tau_longitudinal; centiloid_threshold=nothing)

S = size(tau_matrix, 1)
N = size(tau_matrix, 2)
@assert N == length(roi_names) "ROI name count ($(length(roi_names))) != tau_matrix columns ($N). Check CSV vs loader."

println("Loaded tau_matrix: subjects=$S, rois=$N")
println("Output folders:\n  -> $OUTDIR_SAFE/\n  -> $OUTDIR_EXACT/")


# ROI-wise GMM fitting (across subjects)

keep_roi = trues(N)

# store fitted params for kept ROIs
# For each ROI j: store (μ_non, σ_non, μ_path, σ_path)
μ_non = fill(NaN, N); σ_non = fill(NaN, N)
μ_path = fill(NaN, N); σ_path = fill(NaN, N)

println("\nFitting ROI-wise models (1G vs 2G) with BIC...")

for j in 1:N
    xj = Float64[]
    for i in 1:S
        x = tau_matrix[i, j]
        x === missing && continue
        push!(xj, float(x))
    end

    if length(xj) < 20
        # too little data → exclude (cannot fit reliably)
        keep_roi[j] = false
        continue
    end

    μ1, σ1, ll1, bic1 = fit_1gauss(xj)
    w, a, sa, b, sb, ll2, bic2, conv = fit_2gauss_em(xj)

    # select model by BIC (lower is better)
    if bic1 <= bic2 || !isfinite(bic2)
        keep_roi[j] = false
        continue
    end

    # reorder components by mean: lower = nonpath, higher = path
    if a <= b
        μ_non[j] = a; σ_non[j] = sa
        μ_path[j] = b; σ_path[j] = sb
    else
        μ_non[j] = b; σ_non[j] = sb
        μ_path[j] = a; σ_path[j] = sa
    end
end

kept_idx = findall(keep_roi)
n_kept = length(kept_idx)
println("Kept ROIs (2-Gaussian better): $n_kept / $N")
println("Excluded ROIs (1-Gaussian or insufficient data): $(N - n_kept) / $N")

# save a quick ROI keep/exclude report
roi_report_path = joinpath(OUTDIR_SAFE, "roi_keep_exclude_report.tsv")
open(roi_report_path, "w") do io
    println(io, "roi_index\troi_name\tkept\tmu_non\tsd_non\tmu_path\tsd_path")
    for j in 1:N
        println(io,
            j, "\t", roi_names[j], "\t",
            keep_roi[j] ? "1" : "0", "\t",
            μ_non[j], "\t", σ_non[j], "\t", μ_path[j], "\t", σ_path[j]
        )
    end
end

# also copy to "exact" folder
cp(roi_report_path, joinpath(OUTDIR_EXACT, "roi_keep_exclude_report.tsv"); force=true)

# -----------------------------
# Compute TPI matrix for kept ROIs (subject x ROI, missing preserved)
# -----------------------------
println("\nComputing TPI and within-subject Z-scores...")

TPI = Matrix{Union{Missing,Float64}}(missing, S, N)     # keep full N for easy indexing
TPIz = Matrix{Union{Missing,Float64}}(missing, S, N)

for j in kept_idx
    d_non  = Normal(μ_non[j], max(σ_non[j], EPS))
    d_path = Normal(μ_path[j], max(σ_path[j], EPS))
    for i in 1:S
        x = tau_matrix[i, j]
        x === missing && continue
        TPI[i, j] = tpi_value(float(x), d_non, d_path)
    end
end

# subject-wise zscore across kept ROIs
for i in 1:S
    TPIz[i, :] = zscore_within_subject(vec(TPI[i, :]))
end


# Per-subject plots for visual inspection
plotdir_safe  = joinpath(OUTDIR_SAFE, "per_subject")
plotdir_exact = joinpath(OUTDIR_EXACT, "per_subject")
mkpath(plotdir_safe)
mkpath(plotdir_exact)

summary_path = joinpath(OUTDIR_SAFE, "subject_epicenters_TPIz.tsv")
open(summary_path, "w") do io
    println(io, "subject\tk_epicenters\tepicenter_roi_indices\tepicenter_roi_names")
    for i in 1:S
        sid_raw = subject_IDs[i]
        sid = sanitize_id(sid_raw)

        zrow = vec(TPIz[i, :])
        # only consider kept ROIs for epicenters
        zvals = Float64[]
        ridx  = Int[]
        for j in kept_idx
            zj = zrow[j]
            zj === missing && continue
            push!(zvals, float(zj))
            push!(ridx, j)
        end

        if isempty(zvals)
            # still write a line, but no plot
            println(io, string(sid_raw), "\t0\t\t")
            continue
        end

        # sort by z desc
        ord = sortperm(zvals; rev=true)
        zsorted = zvals[ord]
        jsorted = ridx[ord]
        names_sorted = roi_names[jsorted]

        # epicenters: z > Z_THR
        epi_mask = zsorted .> Z_THR
        epi_js = jsorted[epi_mask]
        epi_names = roi_names[epi_js]
        k_epi = length(epi_js)

        # write summary
        println(io,
            string(sid_raw), "\t",
            k_epi, "\t",
            join(epi_js, ","), "\t",
            join(epi_names, ",")
        )

        # plot: sorted z with threshold; epicenters red
        x = 1:length(zsorted)
        p = scatter(
            x, zsorted;
            xlabel = "ROI rank (sorted by TPI Z)",
            ylabel = "TPI Z-score (within subject)",
            title  = "TPI Z sorted — $(sid_raw)  (epicenters: $k_epi, thr=$Z_THR)",
            markersize = 3,
            legend = false
        )
        hline!(p, [Z_THR]; linestyle=:dash)

        if k_epi > 0
            idx_red = findall(epi_mask)
            scatter!(p, idx_red, zsorted[idx_red]; color=:red, markersize=4)
        end

        outpng = joinpath(plotdir_safe, "TPIz_sorted_$(sid).png")
        savefig(p, outpng)

        # copy into your "exact name" folder too
        cp(outpng, joinpath(plotdir_exact, "TPIz_sorted_$(sid).png"); force=true)
    end
end

cp(summary_path, joinpath(OUTDIR_EXACT, "subject_epicenters_TPIz.tsv"); force=true)

println("\nDone.")
println("Main output:")
println("  ROI report:   $roi_report_path")
println("  Subject list: $summary_path")
println("  Plots:        $plotdir_safe/")
println("Also mirrored to:")
println("  $OUTDIR_EXACT/")
