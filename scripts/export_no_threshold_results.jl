# scripts/export_no_threshold_results.jl
# Export full per-subject results WITHOUT applying epicenter threshold (no plots).
# Reuses GMM ROI decisions/parameters saved by GMM_epicenters.jl.

using DiseaseInitiation
using CSV, DataFrames
using Statistics
using Distributions

const cent_thresh = nothing  # do NOT apply centiloid threshold (include all subjects)

# -----------------------------
# Paths
# -----------------------------
outdir_base = "figures/tau_pet_distrubutions_ren"
gmm_report_path = joinpath(outdir_base, "roi_keep_exclude_report.tsv")

export_dir = joinpath(outdir_base, "exports_no_threshold")
mkpath(export_dir)

@assert isfile(gmm_report_path) """
Cannot find:
  $gmm_report_path

Run scripts/GMM_epicenters.jl first to generate roi_keep_exclude_report.tsv.
"""

# -----------------------------
# Helpers
# -----------------------------
safe_zscore(v::AbstractVector{<:Real}) = begin
    μ = mean(v)
    σ = std(v)
    σ == 0 ? zeros(length(v)) : (v .- μ) ./ σ
end

# Read GMM report robustly (column names may vary slightly)
function read_gmm_report(path::String)
    df = CSV.read(path, DataFrame; delim='\t')

    # Required-ish columns (we try a few common variants)
    function pickcol(possible::Vector{String})
        for c in possible
            if c in names(df)
                return c
            end
        end
        return nothing
    end

    col_roi  = pickcol(["roi", "ROI", "roi_name", "roi_col"])
    col_kept = pickcol(["kept", "keep", "is_kept"])
    col_muN  = pickcol(["mu_non", "μ_non", "mu_nonpath", "mu_non_path"])
    col_sdN  = pickcol(["sd_non", "σ_non", "sd_nonpath", "sd_non_path"])
    col_muP  = pickcol(["mu_path", "μ_path", "mu_pathologic", "mu_pathological"])
    col_sdP  = pickcol(["sd_path", "σ_path", "sd_pathologic", "sd_pathological"])

    # Optional mixing weights (if not present we fall back to 0.5/0.5)
    col_piN  = pickcol(["pi_non", "π_non", "w_non", "weight_non", "mix_non"])
    col_piP  = pickcol(["pi_path", "π_path", "w_path", "weight_path", "mix_path"])

    missing_cols = String[]
    for (nm, col) in [("roi", col_roi), ("kept", col_kept), ("mu_non", col_muN), ("sd_non", col_sdN), ("mu_path", col_muP), ("sd_path", col_sdP)]
        col === nothing && push!(missing_cols, nm)
    end
    isempty(missing_cols) || error("GMM report missing columns: $(join(missing_cols, ", ")). Found: $(names(df))")


    return df, (col_roi=col_roi, col_kept=col_kept, col_muN=col_muN, col_sdN=col_sdN, col_muP=col_muP, col_sdP=col_sdP, col_piN=col_piN, col_piP=col_piP)
end

# Posterior probability of being in pathological component for one observation x
function posterior_path(x::Real; μN::Real, σN::Real, μP::Real, σP::Real, πN::Real=0.5, πP::Real=0.5)
    # guard
    σN = σN == 0 ? 1e-6 : σN
    σP = σP == 0 ? 1e-6 : σP

    nN = pdf(Normal(μN, σN), x)
    nP = pdf(Normal(μP, σP), x)
    denom = πN * nN + πP * nP
    denom == 0 ? 0.0 : (πP * nP) / denom
end

# -----------------------------
# Load dataset (same as mentor scripts)
# -----------------------------
println("\nLoading dataset (no centiloid threshold)...")
FDG_matrix, amyloid_matrix, tau_matrix, subject_IDs =
    load_dataset(:FDG_amyloid_tau_longitudinal; centiloid_threshold=cent_thresh)

FDG_matrix, amyloid_matrix, tau_matrix, nonmissing_subj =
    drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)

subject_IDs = subject_IDs[nonmissing_subj]
S, N = size(tau_matrix)
println("...subjects: $S, ROIs: $N\n")

# -----------------------------
# Load GMM report (ROI decisions + parameters)
# -----------------------------
df_gmm, cols = read_gmm_report(gmm_report_path)

# Map "tau.SUVR.Schaefer200.ROI.idx.X" -> X
function parse_roi_idx(roi_name::AbstractString)
    m = match(r"ROI\.idx\.(\d+)$", roi_name)
    m === nothing ? nothing : parse(Int, m.captures[1])
end

# Build per-ROI parameter tables
kept = falses(N)
μN = fill(NaN, N); σN = fill(NaN, N)
μP = fill(NaN, N); σP = fill(NaN, N)
πN = fill(0.5, N); πP = fill(0.5, N)

n_bad = 0
for r in eachrow(df_gmm)
    roi_name = String(r[cols.col_roi])
    idx = parse_roi_idx(roi_name)
    idx === nothing && (n_bad += 1; continue)
    if 1 <= idx <= N
        kept[idx] = (r[cols.col_kept] == 1)
        μN[idx] = Float64(r[cols.col_muN])
        σN[idx] = Float64(r[cols.col_sdN])
        μP[idx] = Float64(r[cols.col_muP])
        σP[idx] = Float64(r[cols.col_sdP])

        if cols.col_piN !== nothing && cols.col_piP !== nothing
            πN[idx] = Float64(r[cols.col_piN])
            πP[idx] = Float64(r[cols.col_piP])
        end
    end
end

if cols.col_piN === nothing || cols.col_piP === nothing
    println("NOTE: Mixing weights (pi_non/pi_path) not found in GMM report; using 0.5/0.5 for posterior calculation.")
end
n_bad > 0 && println("Warning: $n_bad ROI names could not be parsed into ROI.idx.<n> (ignored).")

kept_inds = findall(kept)
println("Kept ROIs (2-Gaussian by BIC): $(length(kept_inds)) / $N\n")

# -----------------------------
# Compute full per-subject results (NO thresholding)
# -----------------------------
# We export:
# 1) pPath matrix (S x N) - posterior pathological probability per ROI (0..1) for kept ROIs, missing otherwise
# 2) TPIz matrix (S x N)  - within-subject zscore of pPath across kept ROIs, missing otherwise
# 3) Rank matrix (S x N)  - rank within subject among kept ROIs (1 = highest pPath), missing otherwise

pPath = Matrix{Union{Missing,Float64}}(missing, S, N)
TPIz  = Matrix{Union{Missing,Float64}}(missing, S, N)
Rank  = Matrix{Union{Missing,Int}}(missing, S, N)

for i in 1:S
    # tau row might already be Float64 (post drop_missing_rows), but keep it robust
    tau = collect(skipmissing(vec(tau_matrix[i, :])))
    # tau should have length N; if not, skip
    length(tau) == N || continue

    # Compute pPath for kept ROIs
    vals = Float64[]
    idxs = Int[]
    for j in kept_inds
        x = tau[j]
        pp = posterior_path(x; μN=μN[j], σN=σN[j], μP=μP[j], σP=σP[j], πN=πN[j], πP=πP[j])
        pPath[i, j] = pp
        push!(vals, pp)
        push!(idxs, j)
    end

    # Z-score within subject across kept ROIs
    z = safe_zscore(vals)
    for (k, j) in enumerate(idxs)
        TPIz[i, j] = z[k]
    end

    # Rank within subject (1 = highest pPath)
    order = sortperm(vals; rev=true)
    ranks = fill(0, length(vals))
    for (rnk, pos) in enumerate(order)
        ranks[pos] = rnk
    end
    for (k, j) in enumerate(idxs)
        Rank[i, j] = ranks[k]
    end
end

# -----------------------------
# Write wide-format TSVs
# -----------------------------
roi_cols = ["ROI_$j" for j in 1:N]

function write_wide(path::String, mat; subjects=subject_IDs)
    df = DataFrame(subject = String.(subjects))
    for j in 1:N
        df[!, roi_cols[j]] = mat[:, j]
    end
    CSV.write(path, df; delim='\t')
end

pPath_path = joinpath(export_dir, "pPath_allROIs_no_threshold.tsv")
TPIz_path  = joinpath(export_dir, "TPIz_allROIs_no_threshold.tsv")
Rank_path  = joinpath(export_dir, "Rank_allROIs_no_threshold.tsv")

println("Writing exports to:\n  $export_dir\n")
write_wide(pPath_path, pPath)
write_wide(TPIz_path,  TPIz)
write_wide(Rank_path,  Rank)

# -----------------------------
# Also write a compact long-format file (easier to filter/pivot later)
# -----------------------------
long_path = joinpath(export_dir, "long_allROIs_no_threshold.tsv")
df_long = DataFrame(subject=String[], roi_idx=Int[], kept=Int[], tau_suvr=Float64[], pPath=Union{Missing,Float64}[], TPIz=Union{Missing,Float64}[], rank=Union{Missing,Int}[])

for i in 1:S
    tau = collect(skipmissing(vec(tau_matrix[i, :])))
    length(tau) == N || continue
    sid = String(subject_IDs[i])
    for j in 1:N
        push!(df_long, (sid, j, kept[j] ? 1 : 0, tau[j], pPath[i,j], TPIz[i,j], Rank[i,j]))
    end
end

CSV.write(long_path, df_long; delim='\t')

println("Done.")
println("Wide files:")
println("  - $pPath_path")
println("  - $TPIz_path")
println("  - $Rank_path")
println("Long file:")
println("  - $long_path")
