using DiseaseInitiation
using CSV, DataFrames
using Statistics, Distributions
using Plots

const DATA_CSV = "data/ADNI_HABS_amyloid_FDG_longitudinal_tau.csv"
const OUTROOT  = "figures/gmm_qc_max_nonQ_pathQ25"
mkpath(OUTROOT)

const EPS = 1e-9
const MAX_ITERS = 300
const TOL = 1e-7

sanitize_id(s) = replace(string(s), r"[^\w\-]+" => "_")

function fit_2gauss_em(x::Vector{Float64}; max_iters=MAX_ITERS, tol=TOL)
    n = length(x)
    n < 10 && return nothing

    q25, q75 = quantile(x, (0.25, 0.75))
    μ1, μ2 = q25, q75
    σ = std(x) < EPS ? 1.0 : std(x)
    σ1, σ2 = σ, σ
    w = 0.5

    ll_prev = -Inf

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

        σ1 = sqrt(sum(r1 .* (x .- μ1).^2) / s1 + EPS)
        σ2 = sqrt(sum(r2 .* (x .- μ2).^2) / s2 + EPS)

        ll = sum(log.(w .* pdf.(Normal(μ1, σ1), x) .+
                      (1 - w) .* pdf.(Normal(μ2, σ2), x) .+ EPS))

        abs(ll - ll_prev) < tol && break
        ll_prev = ll
    end

    # reorder: left = non-path, right = path
    if μ1 <= μ2
        return (w, μ1, σ1, μ2, σ2)  # w is weight of LEFT component
    else
        return (1 - w, μ2, σ2, μ1, σ1)
    end
end

df_head = CSV.read(DATA_CSV, DataFrame; limit=1)
tau_cols = filter(c -> startswith(c, "tau.SUVR.Schaefer200"), names(df_head))
@assert !isempty(tau_cols) "No tau columns found with prefix tau.SUVR.Schaefer200 in $DATA_CSV"
roi_names = String.(tau_cols)

FDG_matrix, amyloid_matrix, tau_matrix, subject_IDs =
    load_dataset(:FDG_amyloid_tau_longitudinal; centiloid_threshold=nothing)

S, N = size(tau_matrix)
@assert N == length(roi_names) "ROI name count ($(length(roi_names))) != tau_matrix columns ($N)."
println("Loaded tau matrix: $S subjects × $N ROIs")

# yellow-line quantiles 
const NON_Q_LIST = [0.95, 0.975, 0.99, 0.999]  # yellow line = Q(non)
const PATH_Q_LOW = 0.25                        # green line = Q25(path)

# 4 variants 
for non_q in NON_Q_LIST
    qtag = replace(string(non_q), "." => "p")  # 0.975 -> 0p975
    outdir = joinpath(OUTROOT, "nonQ_" * qtag * "_pathQ25")
    mkpath(outdir)

    plotdir = joinpath(outdir, "plots")
    mkpath(plotdir)

    println("\n=== Running variant: nonQ=$(non_q), pathQ=$(PATH_Q_LOW) ===")
    println("Output: $outdir")

    # per-subject path summary output
    subj_summary = DataFrame(
        subject = String[],
        k_path  = Int[],
        path_roi_indices = String[],
        path_roi_names   = String[]
    )

    # collect per-subject lists while looping ROIs (use vector-of-vectors)
    path_lists_idx = [Int[] for _ in 1:S]

    # ROI-level threshold table (optional but useful)
    roi_thresholds = DataFrame(
        roi_index = Int[],
        roi_name  = String[],
        n         = Int[],
        w_non     = Float64[],
        mu_non    = Float64[],
        sd_non    = Float64[],
        mu_path   = Float64[],
        sd_path   = Float64[],
        L_nonQ    = Float64[],
        G_pathQ25 = Float64[],
        T_start   = Float64[],
        n_nonpath = Int[],
        n_path    = Int[]
    )


    for j in 1:N
        # collect non-missing tau values across subjects for this ROI
        x = Float64[]
        for i in 1:S
            v = tau_matrix[i, j]
            v === missing && continue
            push!(x, float(v))
        end
        length(x) < 30 && continue

        params = fit_2gauss_em(x)
        params === nothing && continue

        w_non, μ_non, σ_non, μ_path, σ_path = params
        d_non  = Normal(μ_non, max(σ_non, EPS))
        d_path = Normal(μ_path, max(σ_path, EPS))

        # Yellow + Green
        L = quantile(d_non, non_q)         # yellow
        G = quantile(d_path, PATH_Q_LOW)   # green

        # Start threshold: whichever is larger (more conservative)
        T = max(L, G)

        # classify subjects for this ROI + count
        n_non = 0
        n_pat = 0

        for i in 1:S
            v = tau_matrix[i, j]
            v === missing && continue
            xv = float(v)

            if xv > T
                n_pat += 1
                push!(path_lists_idx[i], j)
            else
                n_non += 1
            end
        end

        push!(roi_thresholds, (j, roi_names[j], length(x),
                               w_non, μ_non, σ_non, μ_path, σ_path,
                               L, G, T, n_non, n_pat))

        # ---- Plot ----
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
            title="ROI $(j): $(roi_names[j]) | L=Q$(non_q)(non), G=Q25(path), T=max(L,G)"
        )

        plot!(p, xs, pdf_non;  label="Non-path (Left)", linewidth=2)
        plot!(p, xs, pdf_path; label="Path (Right)", linewidth=2)
        plot!(p, xs, pdf_mix;  label="Mixture", linewidth=2, linestyle=:dash)

        # lines: yellow L, green G, and final threshold T
        vline!(p, [L]; linewidth=2, linestyle=:dot,      label="Yellow L = Q$(non_q)(non)")
        vline!(p, [G]; linewidth=2, linestyle=:dashdot,  label="Green  G = Q25(path)")
        vline!(p, [T]; linewidth=3, linestyle=:solid,    label="Threshold T = max(L,G)")

        savefig(p, joinpath(plotdir, "ROI_$(j)_GMM_QC_nonQ_$(qtag).png"))
    end


    for i in 1:S
        sid_raw = string(subject_IDs[i])
        idxs = path_lists_idx[i]
        unique!(idxs)
        sort!(idxs)
        names = roi_names[idxs]

        push!(subj_summary, (
            sid_raw,
            length(idxs),
            join(idxs, ","),
            join(names, ",")
        ))
    end

    # ---- Write outputs ----
    thresholds_path = joinpath(outdir, "roi_thresholds_nonQ_" * qtag * ".tsv")
    subj_path       = joinpath(outdir, "subject_path_regions_nonQ_" * qtag * ".tsv")

    CSV.write(thresholds_path, roi_thresholds; delim='\t')
    CSV.write(subj_path, subj_summary; delim='\t')

    println("Saved plots:     $plotdir/ROI_*")
    println("Saved ROI table: $thresholds_path")
    println("Saved subject summary: $subj_path")
end

println("\nAll variants done.")
println("Root output folder: $OUTROOT")