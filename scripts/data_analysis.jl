using DiseaseInitiation
using Plots
using Statistics
using StatsBase  # median, mad, quantile

const cent_thresh = nothing  # centiloid threshold for amyloid positivity

# -----------------------------
# Outlier detectors (return indices into the ORIGINAL tau vector)
# -----------------------------
function outliers_mad(tau::AbstractVector{<:Real}; zthr::Real = 3.5, eps::Real = 1e-12)
    med = median(tau)
    m = mad(tau; normalize=false)  # robust scale
    m = m < eps ? eps : m
    z = (tau .- med) ./ m
    return findall(>(zthr), z)
end

function outliers_iqr(tau::AbstractVector{<:Real}; k::Real = 1.5)
    q1, q3 = quantile(tau, (0.25, 0.75))
    iqr = q3 - q1
    thresh = q3 + k * iqr
    return findall(>(thresh), tau)
end

"""
Rank-gap / elbow detector.
Find k = argmax gap among top `topK_search` ranks; return top-k indices.
This returns a *set of top-ranked ROIs*, not necessarily "statistical" outliers.
"""
function outliers_rankgap(tau::AbstractVector{<:Real}; topK_search::Int = 20, kmax::Int = 20)
    N = length(tau)
    K = min(topK_search, N-1, kmax)
    ord = sortperm(tau; rev=true)
    s = tau[ord]
    gaps = s[1:K] .- s[2:K+1]
    k = argmax(gaps)  # 1..K
    return ord[1:k]
end

# -----------------------------
# Helper: map outlier ROI indices -> positions in sorted plot
# -----------------------------
function sorted_positions_from_indices(ord::AbstractVector{Int}, out_idx::AbstractVector{Int})
    # ord: ROI indices in sorted order (length N)
    # inv: inverse permutation mapping ROI index -> sorted position
    inv = invperm(ord)
    return sort(inv[out_idx])  # sorted positions (1..N)
end


# -----------------------------
# Load data
# -----------------------------
println("\nloading dataset...")
FDG_matrix, amyloid_matrix, tau_matrix, subject_IDs =
    load_dataset(:FDG_amyloid_tau_longitudinal; centiloid_threshold=cent_thresh)
println("...using $(size(FDG_matrix,1)) subjects.\n")

FDG_matrix, amyloid_matrix, tau_matrix, nonmissing_subj =
    drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)
subject_IDs = subject_IDs[nonmissing_subj]

# -----------------------------
# Output folder
# -----------------------------
outdir = "figures/tau_PET_distributions"
mkpath(outdir)

# -----------------------------
# One set of plots per subject
# -----------------------------
S = size(tau_matrix, 1)

for i in 1:S
    # tau should already be Float64 after drop_missing_rows, but keep it robust
    tau = collect(skipmissing(vec(tau_matrix[i, :])))
    isempty(tau) && continue

    N = length(tau)
    sid = replace(string(subject_IDs[i]), r"[^\w\-]+" => "_")

    # Sort for strip plot
    ord = sortperm(tau)              # ascending
    tau_sorted = tau[ord]

    # --- detect outliers in ORIGINAL indexing ---
    idx_mad  = outliers_mad(tau; zthr=3.5)
    idx_iqr  = outliers_iqr(tau; k=1.5)
    idx_gap  = outliers_rankgap(tau; topK_search=20, kmax=20)

    # --- map to sorted positions for plotting ---
    pos_mad = sorted_positions_from_indices(ord, idx_mad)
    pos_iqr = sorted_positions_from_indices(ord, idx_iqr)
    pos_gap = sorted_positions_from_indices(ord, idx_gap)

    # Convenience for plotting
    x = 1:N

    # -----------------------------
    # Histogram (optional; keep if you still want it)
    # -----------------------------
    p_hist = histogram(
        tau;
        bins=:auto,
        xlabel="Tau PET (SUVR)",
        ylabel="Count",
        title="Tau PET histogram — $(subject_IDs[i])",
        legend=false
    )
    vline!(p_hist, [mean(tau)]; linestyle=:dash, label="")
    #savefig(p_hist, joinpath(outdir, "tau_hist_$(sid).png"))

    # -----------------------------
    # Base strip plot (sorted values)
    # -----------------------------
    function plot_with_outliers(tau_sorted, pos_out, label::String)
        p = scatter(
            x, tau_sorted;
            xlabel="ROI rank (sorted)",
            ylabel="Tau PET (SUVR)",
            title="Tau PET sorted — $(subject_IDs[i])  [$label]",
            markersize=3,
            legend=false
        )
        if !isempty(pos_out)
            scatter!(p, pos_out, tau_sorted[pos_out]; markersize=4, color=:red, label=false)
        end
        return p
    end

    p_mad = plot_with_outliers(tau_sorted, pos_mad, "MAD z>3.5 (n=$(length(idx_mad)))")
    savefig(p_mad, joinpath(outdir, "tau_strip_outliers_$(sid)_MAD.png"))

    p_iqr = plot_with_outliers(tau_sorted, pos_iqr, "IQR k=1.5 (n=$(length(idx_iqr)))")
    savefig(p_iqr, joinpath(outdir, "tau_strip_outliers_$(sid)_IQR.png"))

    p_gap = plot_with_outliers(tau_sorted, pos_gap, "Rank-gap top-k (k=$(length(idx_gap)))")
    savefig(p_gap, joinpath(outdir, "tau_strip_outliers_$(sid)_RANKGAP.png"))
end

println("Saved tau distribution plots (hist + 3 outlier strip plots) to: $outdir/")
