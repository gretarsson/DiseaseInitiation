using DiseaseInitiation
using CSV, DataFrames
using Statistics, StatsBase
using Distributions
using Plots

# ========= PATHS =========
const DATA_CSV = "data/ADNI_HABS_amyloid_FDG_longitudinal_tau.csv"
const OUTDIR   = "figures/tau_suvr_hist_bic_first5kept"
mkpath(OUTDIR)

# ========= CONSTANTS =========
const EPS = 1e-9
const MAX_ITERS = 300
const TOL = 1e-7

# ========= 1-Gaussian (for BIC) =========
function fit_1gauss(x::Vector{Float64})
    n = length(x)
    μ = mean(x)
    σ = std(x)
    σ = (σ < EPS) ? EPS : σ
    ll = sum(logpdf.(Normal(μ, σ), x))
    p = 2
    bic = -2 * ll + p * log(n)
    return μ, σ, ll, bic
end

# ========= 2-Gaussian EM (for BIC + plotting) =========
function fit_2gauss_em(x::Vector{Float64}; max_iters=MAX_ITERS, tol=TOL)
    n = length(x)
    n < 10 && return (0.5, mean(x), std(x)+EPS, mean(x), std(x)+EPS, -Inf, Inf, false)

    q25, q75 = quantile(x, (0.25, 0.75))
    μ1, μ2 = q25, q75
    σ0 = std(x)
    σ1 = (σ0 < EPS) ? 1.0 : σ0
    σ2 = σ1
    w = 0.5

    ll_prev = -Inf
    converged = false

    for _ in 1:max_iters
        d1 = Normal(μ1, max(σ1, EPS))
        d2 = Normal(μ2, max(σ2, EPS))

        p1 = w .* pdf.(d1, x)
        p2 = (1 - w) .* pdf.(d2, x)
        denom = p1 .+ p2 .+ EPS

        r1 = p1 ./ denom
        r2 = 1 .- r1

        w = clamp(mean(r1), 1e-3, 1 - 1e-3)
        s1 = sum(r1) + EPS
        s2 = sum(r2) + EPS

        μ1 = sum(r1 .* x) / s1
        μ2 = sum(r2 .* x) / s2

        v1 = sum(r1 .* (x .- μ1).^2) / s1
        v2 = sum(r2 .* (x .- μ2).^2) / s2
        σ1 = sqrt(max(v1, EPS))
        σ2 = sqrt(max(v2, EPS))

        ll = sum(log.(w .* pdf.(Normal(μ1, σ1), x) .+
                      (1 - w) .* pdf.(Normal(μ2, σ2), x) .+ EPS))

        if abs(ll - ll_prev) < tol * (1 + abs(ll_prev))
            converged = true
            ll_prev = ll
            break
        end
        ll_prev = ll
    end

    ll = ll_prev
    p = 5
    bic = -2 * ll + p * log(n)
    return w, μ1, σ1, μ2, σ2, ll, bic, converged
end

# ========= Load ROI names =========
df_head = CSV.read(DATA_CSV, DataFrame; limit=1)
tau_cols = filter(c -> startswith(c, "tau.SUVR.Schaefer200"), names(df_head))
@assert !isempty(tau_cols) "No tau columns found with prefix tau.SUVR.Schaefer200 in $DATA_CSV"
roi_names = String.(tau_cols)

# ========= Load dataset =========
FDG_matrix, amyloid_matrix, tau_matrix, subject_IDs =
    load_dataset(:FDG_amyloid_tau_longitudinal; centiloid_threshold=nothing)

S, N = size(tau_matrix)
@assert N == length(roi_names) "ROI name count ($(length(roi_names))) != tau_matrix columns ($N)."

println("Loaded tau matrix: $S subjects × $N ROIs")
println("Selecting ROIs by BIC (2G vs 1G), then plotting first 5 kept...")

# ========= BIC selection =========
report = DataFrame(
    roi_index = Int[],
    roi_name  = String[],
    n         = Int[],
    bic_1g    = Float64[],
    bic_2g    = Float64[],
    kept_2g   = Int[],
    w         = Float64[],
    mu_non    = Float64[],
    sd_non    = Float64[],
    mu_path   = Float64[],
    sd_path   = Float64[]
)

kept_list = Int[]

for j in 1:N
    x = Float64[]
    for i in 1:S
        v = tau_matrix[i, j]
        v === missing && continue
        push!(x, float(v))
    end

    if length(x) < 30
        push!(report, (j, roi_names[j], length(x), NaN, NaN, 0, NaN, NaN, NaN, NaN, NaN))
        continue
    end

    _, _, _, bic1 = fit_1gauss(x)
    w, a, sa, b, sb, _, bic2, _ = fit_2gauss_em(x)

    kept = (isfinite(bic2) && bic2 < bic1) ? 1 : 0
    if kept == 1
        push!(kept_list, j)
    end

    # reorder by mean for reporting (non=left, path=right)
    if a <= b
        mu_non, sd_non, mu_path, sd_path = a, sa, b, sb
        w_non = w
    else
        mu_non, sd_non, mu_path, sd_path = b, sb, a, sa
        w_non = 1 - w
    end

    push!(report, (j, roi_names[j], length(x), bic1, bic2, kept, w_non, mu_non, sd_non, mu_path, sd_path))
end

# save report
report_path = joinpath(OUTDIR, "bic_keep_report.tsv")
CSV.write(report_path, report; delim='\t')
println("Saved BIC report: $report_path")
println("Kept by BIC (2G better): $(length(kept_list)) / $N")

# ========= Plot first 5 kept ROIs =========
n_show = min(5, length(kept_list))
if n_show == 0
    println("No ROIs passed BIC (2G < 1G). Nothing to plot.")
    exit()
end

for idx in 1:n_show
    j = kept_list[idx]

    # collect x again
    x = Float64[]
    for i in 1:S
        v = tau_matrix[i, j]
        v === missing && continue
        push!(x, float(v))
    end

    # refit to get parameters (could also reuse, but keep simple/robust)
    _, _, _, bic1 = fit_1gauss(x)
    w, a, sa, b, sb, _, bic2, _ = fit_2gauss_em(x)

    # reorder by mean: left=non, right=path
    if a <= b
        μ_non, σ_non, μ_path, σ_path = a, sa, b, sb
        w_non = w
    else
        μ_non, σ_non, μ_path, σ_path = b, sb, a, sa
        w_non = 1 - w
    end

    d_non  = Normal(μ_non, max(σ_non, EPS))
    d_path = Normal(μ_path, max(σ_path, EPS))

    xmin = minimum(x)
    xmax = maximum(x)
    xs = range(xmin, xmax; length=600)

    pdf_non  = w_non .* pdf.(d_non, xs)
    pdf_path = (1 - w_non) .* pdf.(d_path, xs)
    pdf_mix  = pdf_non .+ pdf_path

    p = histogram(
        x;
        bins=40,
        normalize=true,
        alpha=0.4,
        label="Data",
        xlabel="Tau SUVR",
        ylabel="Density",
        title="ROI $(j): $(roi_names[j])  |  BIC1=$(round(bic1, digits=2))  BIC2=$(round(bic2, digits=2))"
    )

    plot!(p, xs, pdf_non;  linewidth=2, label="Non-path (Left)")
    plot!(p, xs, pdf_path; linewidth=2, label="Path (Right)")
    plot!(p, xs, pdf_mix;  linewidth=2, linestyle=:dash, label="Mixture")

    outpng = joinpath(OUTDIR, "ROI_$(j)_tauSUVR_hist_BICkept.png")
    savefig(p, outpng)
    println("Saved: $outpng")
end

println("Done. Output folder: $OUTDIR")
