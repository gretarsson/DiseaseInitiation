#!/usr/bin/env julia
# -*- coding: utf-8 -*-

using LinearAlgebra, Statistics
using CSV, DataFrames
using DiseaseInitiation  # your module

# -----------------------------
# CONFIG
# -----------------------------
const IN_CSV          = "data/ADNI_HABS_amyloid_FDG_longitudinal_tau.csv"
const OUT_CSV         = "data/ADNI_HABS_amyloid_FDG_longitudinal_tau_with_epi_hazard.csv"
const OUT_GROUPAVG_CSV = "data/ADNI_HABS_groupavg_amyloid_FDG_epi_hazard.csv"
const W_CSV           = "data/Schaefer2018_200Parcels_CN.csv"

const TSPAN = (0.0, 20.0)
const TN    = 100
const ρ     = 1.0
const k     = 0.0
const λ     = 0.0

# Column prefixes in your CSV
const FDG_PREFIX = "FDG.SUVR.Schaefer200"
const AMY_PREFIX = "centiloid.amyloid.SUVR.Schaefer200"

# New column prefixes we will create (200 columns per vector)
const EPI_FDG_POS_PREFIX = "epi_hazard.FDG.pos.Schaefer200"
const EPI_FDG_NEG_PREFIX = "epi_hazard.FDG.neg.Schaefer200"
const EPI_AMY_POS_PREFIX = "epi_hazard.amyloid.pos.Schaefer200"
const EPI_AMY_NEG_PREFIX = "epi_hazard.amyloid.neg.Schaefer200"

# -----------------------------
# Helpers
# -----------------------------
# Normalize a vector to [0,1]. If constant, return zeros.
function normalize01(v::AbstractVector{<:Real})
    vmin = minimum(v)
    vmax = maximum(v)
    denom = vmax - vmin
    if denom == 0
        return zeros(Float64, length(v))
    else
        return (Float64.(v) .- vmin) ./ denom
    end
end

# Make names like "...Schaefer200.001" ... ".200"
function roi_cols_with_prefix(prefix::AbstractString, N::Int)
    return [Symbol(prefix * "." * lpad(string(i), 3, '0')) for i in 1:N]
end

# Read ROI vector from a row for given columns; returns Vector{Union{Missing,Float64}}
function get_row_vector(df::DataFrame, row::Int, cols::Vector{Symbol})
    return Vector{Union{Missing,Float64}}(df[row, cols])
end

# True if fully observed (no missings)
is_fully_observed(v) = !any(ismissing, v)

# Compute mean ROI vector across rows that are fully observed for the given columns
function mean_vector_across_complete_rows(df::DataFrame, cols::Vector{Symbol})
    complete_rows = Int[]
    for r in 1:nrow(df)
        v = get_row_vector(df, r, cols)
        if is_fully_observed(v)
            push!(complete_rows, r)
        end
    end

    if isempty(complete_rows)
        error("No fully observed rows found for columns with prefix: $(String(cols[1]))")
    end

    X = Matrix{Float64}(undef, length(complete_rows), length(cols))
    for (i, r) in enumerate(complete_rows)
        X[i, :] = Float64.(get_row_vector(df, r, cols))
    end

    return vec(mean(X, dims=1)), complete_rows
end

# -----------------------------
# Load adjacency and build Laplacian
# -----------------------------
W = Matrix(CSV.read(W_CSV, DataFrame; header=false))
N = size(W, 1)
@assert N == 200 "Expected 200 ROIs (Schaefer200). Got N = $N"

L = DiseaseInitiation.laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))  # match your normalization

# -----------------------------
# Load CSV and clean "NA" strings -> missing
# -----------------------------
df = CSV.read(IN_CSV, DataFrame)

for c in names(df)
    col = df[!, c]
    if eltype(col) <: AbstractString
        df[!, c] = allowmissing(col)
        replace!(df[!, c], "NA" => missing)
    end
end

# Identify ROI columns
FDG_cols = Symbol.(filter(c -> startswith(c, FDG_PREFIX), String.(names(df))))
AMY_cols = Symbol.(filter(c -> startswith(c, AMY_PREFIX), String.(names(df))))

@assert length(FDG_cols) == N "Found $(length(FDG_cols)) FDG ROI columns; expected $N."
@assert length(AMY_cols) == N "Found $(length(AMY_cols)) amyloid ROI columns; expected $N."

# Parse ROI columns to Union{Missing,Float64}
for c in vcat(FDG_cols, AMY_cols)
    df[!, c] = passmissing(x -> x isa Real ? Float64(x) : parse(Float64, x)).(df[!, c])
end

# -----------------------------
# Pre-create output columns (4 * 200), initialize as missing
# -----------------------------
epi_fdg_pos_cols = roi_cols_with_prefix(EPI_FDG_POS_PREFIX, N)
epi_fdg_neg_cols = roi_cols_with_prefix(EPI_FDG_NEG_PREFIX, N)
epi_amy_pos_cols = roi_cols_with_prefix(EPI_AMY_POS_PREFIX, N)
epi_amy_neg_cols = roi_cols_with_prefix(EPI_AMY_NEG_PREFIX, N)

for cols in (epi_fdg_pos_cols, epi_fdg_neg_cols, epi_amy_pos_cols, epi_amy_neg_cols)
    for c in cols
        if !(c in names(df))
            df[!, c] = Vector{Union{Missing,Float64}}(missing, nrow(df))
        end
    end
end

# -----------------------------
# Main loop: per row predictions
# -----------------------------
u0 = ones(N)
zero_M = diagm(zeros(N))

for r in 1:nrow(df)
    # ---- FDG ----
    fdg_raw = get_row_vector(df, r, FDG_cols)
    if is_fully_observed(fdg_raw)
        fdg = normalize01(Float64.(fdg_raw))
        fdg_M = diagm(fdg)

        # Positive FDG impact: eps2 = +1, amyloid zero
        Upos = DiseaseInitiation.disease_initiation_timeseries(
            L, zero_M, fdg_M, u0, ρ, 0.0, +1.0, k, λ, TSPAN, TN
        )[:, end]

        # Negative FDG impact: eps2 = -1, amyloid zero
        Uneg = DiseaseInitiation.disease_initiation_timeseries(
            L, zero_M, fdg_M, u0, ρ, 0.0, -1.0, k, λ, TSPAN, TN
        )[:, end]

        df[r, epi_fdg_pos_cols] = Upos
        df[r, epi_fdg_neg_cols] = Uneg
    end

    # ---- Amyloid (centiloid ROI vector) ----
    amy_raw = get_row_vector(df, r, AMY_cols)
    if is_fully_observed(amy_raw)
        amy = normalize01(Float64.(amy_raw))
        amy_M = diagm(amy)

        # Positive amyloid impact: eps1 = +1, FDG zero
        Upos = DiseaseInitiation.disease_initiation_timeseries(
            L, amy_M, zero_M, u0, ρ, +1.0, 0.0, k, λ, TSPAN, TN
        )[:, end]

        # Negative amyloid impact: eps1 = -1, FDG zero
        Uneg = DiseaseInitiation.disease_initiation_timeseries(
            L, amy_M, zero_M, u0, ρ, -1.0, 0.0, k, λ, TSPAN, TN
        )[:, end]

        df[r, epi_amy_pos_cols] = Upos
        df[r, epi_amy_neg_cols] = Uneg
    end
end

# -----------------------------
# Group-average predictions:
# average scans first, then predict once
# -----------------------------
fdg_mean_raw, fdg_complete_rows = mean_vector_across_complete_rows(df, FDG_cols)
amy_mean_raw, amy_complete_rows = mean_vector_across_complete_rows(df, AMY_cols)

fdg_mean = normalize01(fdg_mean_raw)
amy_mean = normalize01(amy_mean_raw)

fdg_mean_M = diagm(fdg_mean)
amy_mean_M = diagm(amy_mean)

# FDG group-average predictions
fdg_group_pos = DiseaseInitiation.disease_initiation_timeseries(
    L, zero_M, fdg_mean_M, u0, ρ, 0.0, +1.0, k, λ, TSPAN, TN
)[:, end]

fdg_group_neg = DiseaseInitiation.disease_initiation_timeseries(
    L, zero_M, fdg_mean_M, u0, ρ, 0.0, -1.0, k, λ, TSPAN, TN
)[:, end]

# Amyloid group-average predictions
amy_group_pos = DiseaseInitiation.disease_initiation_timeseries(
    L, amy_mean_M, zero_M, u0, ρ, +1.0, 0.0, k, λ, TSPAN, TN
)[:, end]

amy_group_neg = DiseaseInitiation.disease_initiation_timeseries(
    L, amy_mean_M, zero_M, u0, ρ, -1.0, 0.0, k, λ, TSPAN, TN
)[:, end]

group_df = DataFrame(
    ROI = 1:N,
    FDG_mean_raw = fdg_mean_raw,
    FDG_mean_norm = fdg_mean,
    amyloid_mean_raw = amy_mean_raw,
    amyloid_mean_norm = amy_mean,
    epi_hazard_FDG_pos = fdg_group_pos,
    epi_hazard_FDG_neg = fdg_group_neg,
    epi_hazard_amyloid_pos = amy_group_pos,
    epi_hazard_amyloid_neg = amy_group_neg,
)

# -----------------------------
# Save
# -----------------------------
CSV.write(OUT_CSV, df; missingstring="NA")
println("Wrote individualized predictions CSV to:\n  $OUT_CSV")

CSV.write(OUT_GROUPAVG_CSV, group_df)
println("Wrote group-average predictions CSV to:\n  $OUT_GROUPAVG_CSV")

println("FDG group average used $(length(fdg_complete_rows)) complete rows.")
println("Amyloid group average used $(length(amy_complete_rows)) complete rows.")