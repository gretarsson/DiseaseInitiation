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
    if N ≤ 1
        return Int[]
    end
    K = min(topK_search, N - 1, kmax)
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
    inv = invperm(ord)          # ROI index -> sorted position
    return sort(inv[out_idx])   # sorted positions (1..N)
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
# Across-the-board epicenter tracking (methods may disagree)
# Keep subjects ONLY if each method returns ≥1 epicenter ROI.
# -----------------------------
subjects_all = String[]
rois_mad_all = Vector{Vector{Int}}()
rois_iqr_all = Vector{Vector{Int}}()
rois_gap_all = Vector{Vector{Int}}()

# -----------------------------
# One set of plots per subject
# -----------------------------
S = size(tau_matrix, 1)

for i in 1:S
    # tau should already be Float64 after drop_missing_rows, but keep it robust
    tau = collect(skipmissing(vec(tau_matrix[i, :])))
    isempty(tau) && continue

    N = length(tau)
    sid_raw = string(subject_IDs[i])
    sid = replace(sid_raw, r"[^\w\-]+" => "_")

    # Sort for strip plot
    ord = sortperm(tau)              # ascending
    tau_sorted = tau[ord]

    # --- detect epicenters/outliers in ORIGINAL indexing ---
    idx_mad = outliers_mad(tau; zthr=3.5)
    idx_iqr = outliers_iqr(tau; k=1.5)
    idx_gap = outliers_rankgap(tau; topK_search=20, kmax=20)

    # --- keep subject only if ALL methods produced ≥1 epicenter ---
    ok_all = !isempty(idx_mad) && !isempty(idx_iqr) && !isempty(idx_gap)
    if ok_all
        push!(subjects_all, sid_raw)
        push!(rois_mad_all, sort(idx_mad))
        push!(rois_iqr_all, sort(idx_iqr))
        push!(rois_gap_all, sort(idx_gap))
    end

    # --- map to sorted positions for plotting ---
    pos_mad = sorted_positions_from_indices(ord, idx_mad)
    pos_iqr = sorted_positions_from_indices(ord, idx_iqr)
    pos_gap = sorted_positions_from_indices(ord, idx_gap)

    x = 1:N

    # -----------------------------
    # Histogram (optional)
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
    # savefig(p_hist, joinpath(outdir, "tau_hist_$(sid).png"))

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

# -----------------------------
# Write across-the-board subjects to txt (methods may disagree)
# -----------------------------
txt_path = joinpath(outdir, "subjects_with_epicenters_ALL_metrics.txt")
open(txt_path, "w") do io
    println(io, "# Subjects for which ALL metrics return ≥1 epicenter ROI (methods may disagree).")
    println(io, "# Format:")
    println(io, "# SubjectID<TAB>MAD_ROIs<TAB>IQR_ROIs<TAB>RANKGAP_ROIs<TAB>Intersection<TAB>Union")
    println(io)

    for k in eachindex(subjects_all)
        sid = subjects_all[k]
        mad_rois = rois_mad_all[k]
        iqr_rois = rois_iqr_all[k]
        gap_rois = rois_gap_all[k]

        inter = sort(collect(intersect(mad_rois, iqr_rois, gap_rois)))
        uni   = sort(collect(union(mad_rois, iqr_rois, gap_rois)))

        println(io,
            sid, "\t",
            join(mad_rois, ","), "\t",
            join(iqr_rois, ","), "\t",
            join(gap_rois, ","), "\t",
            join(inter, ","), "\t",
            join(uni, ",")
        )
    end
end

println("Saved tau distribution plots to: $outdir/")
println("Saved across-the-board epicenter list to: $txt_path")
