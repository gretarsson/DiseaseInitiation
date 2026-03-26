using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation
Random.seed!(1234)
using CSV, DataFrames, Statistics, Dates
default(dpi=300)

function enforce_positive_eps(epsA::Float64, epsF::Float64,
                              a::AbstractVector{<:Real}, f::AbstractVector{<:Real};
                              δ::Float64 = 1e-6)
    q = 1 .+ epsA .* a .+ epsF .* f
    m = minimum(q)
    if m >= δ
        return epsA, epsF, q
    end

    # Need to increase q by Δ = (δ - m) > 0.
    Δ = δ - m

    # Decide which epsilon to adjust: prefer the biomarker with larger leverage (max value).
    maxA = maximum(a)
    maxF = maximum(f)

    if maxF > 0 && (maxF >= maxA || maxA == 0)
        # increase epsF so that epsF*f can lift the worst-case region
        epsF_new = epsF + Δ / maxF
        q_new = 1 .+ epsA .* a .+ epsF_new .* f
        return epsA, epsF_new, q_new
    elseif maxA > 0
        # otherwise increase epsA
        epsA_new = epsA + Δ / maxA
        q_new = 1 .+ epsA_new .* a .+ epsF .* f
        return epsA_new, epsF, q_new
    else
        # both biomarkers are all-zero; only option is to rely on baseline 1
        # (shouldn't happen if you normalized to [0,1] with variation)
        return epsA, epsF, q
    end
end


W = Matrix(CSV.read("data/Schaefer2018_200Parcels_CN.csv", DataFrame; header=false))
N = size(W, 1)

data = load_dataset(:FDG_amyloid_tau_longitudinal)
if length(data) == 3
    FDG_matrix, amyloid_matrix, tau_matrix = data
elseif length(data) == 4
    FDG_matrix, amyloid_matrix, tau_matrix, subjects = data
else
    error("Unexpected number of outputs from load_dataset: $(length(data))")
end

FDG_matrix, amyloid_matrix, tau_matrix = drop_missing_rows(FDG_matrix, amyloid_matrix, tau_matrix)

# normalize
if FDG_matrix !== nothing
    for i in axes(FDG_matrix, 1)
        row = FDG_matrix[i, :]
        FDG_matrix[i, :] .= (row .- minimum(row)) ./ (maximum(row) - minimum(row) + eps())  # eps() avoid 0/0
    end
end

if amyloid_matrix !== nothing
    for i in axes(amyloid_matrix, 1)
        row = amyloid_matrix[i, :]
        amyloid_matrix[i, :] .= (row .- minimum(row)) ./ (maximum(row) - minimum(row) + eps())
    end
end

# the smae
L = laplacian(W, kind=:out)
L = L ./ maximum(eigvals(L))   # normalize Laplacian

tspan = (0.0, 20.0)
Tn    = 100

ρ    = 1.0
epsA = -0.6     
epsF = -0.6     
k    = 0.0
λ    = 0.0

outdir = "figures/timeseries_subjects_FDG_positive"
isdir(outdir) || mkpath(outdir)

t = range(tspan[1], tspan[2], length=Tn)

for i in axes(amyloid_matrix, 1)
    a = vec(amyloid_matrix[i, :])
    f = vec(FDG_matrix[i, :])

    # enforce q>=δ for THIS subject (so generator stays Metzler)
    epsA_i, epsF_i, q = enforce_positive_eps(epsA, epsF, a, f; δ=1e-6)

    amyloid_M = diagm(a)
    FDG_M     = diagm(f)

    init_timeseries = disease_initiation_timeseries(
        L, amyloid_M, FDG_M, ones(N),
        ρ, epsA_i, epsF_i, k, λ,
        tspan, Tn
    )

    # quick sanity check: u(t) should be >= 0 (numerical noise tolerance)
    if minimum(init_timeseries) < -1e-10
        @warn "Negative values detected (numerical?)" subject=i minval=minimum(init_timeseries) epsA=epsA_i epsF=epsF_i
    end

    plot(t, init_timeseries',
        xlabel="Time",
        ylabel="u(t)",
        title="Subject $(i) — Positive (q≥0 enforced)",
        lw=2,
        alpha=0.6,
        legend=false)

    savefig(joinpath(outdir, "timeseries_$(i).png"))
end

println("Done. Saved figures to: $(outdir)")
