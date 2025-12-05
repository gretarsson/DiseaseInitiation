module DiseaseInitiation

export laplacian, disease_initiation_vector, disease_initiation_matrix, epicenter_accuracy, load_dataset, drop_missing_rows
export disease_initiation_timeseries
export spearman_with_pvalue
export normalize_rows
export make_objective_timesweep, make_objective_global
export save_optimization_summary

using DifferentialEquations, LinearAlgebra, Dates, CSV, DataFrames, StatsBase, Distributions


function make_disease_initiation(L, M)
    function f!(du, u, p, t)
        ρ, ϵ, k, λ = p
        du .= -ρ * L * (I + ϵ .* M) * u .+ k .- λ .* u
    end
    return f!
end

function laplacian(A; kind::Symbol = :out, normalize::Bool = false)
    size(A,1) == size(A,2) ||
        throw(ArgumentError("Adjacency matrix must be square"))

    if kind == :out
        d = sum(A, dims=2)            # row sums
    elseif kind == :in
        d = sum(A, dims=1)'           # column sums
    else
        throw(ArgumentError("kind must be :out or :in"))
    end

    D = Diagonal(vec(d))
    L = D - A
    if normalize
        L = L ./ maximum(eigvals(L))
    end
    return L
end


function disease_initiation_vector(L, M1, M2, u0, ρ, ϵ1, ϵ2, k, λ, T)
    N = length(u0)

    # Base linear operator X
    X = -ρ * L * (I + ϵ1 * M1 + ϵ2 * M2) - λ * I

    # Constant forcing term b = k * ones
    b = k .* ones(N)

    # Build augmented matrix
    Xtilde = zeros(N+1, N+1)
    Xtilde[1:N, 1:N] = X
    Xtilde[1:N, N+1] = b
    # last row remains zeros (dummy variable)

    # Augmented initial condition
    utilde0 = vcat(u0, 1.0)

    # Solve via matrix exponential
    utilde_T = exp(Xtilde * T) * utilde0

    # Return only physical variables (discard dummy)
    return utilde_T[1:N]
end


function disease_initiation_timeseries(L, M1, M2, u0, ρ, ϵ1, ϵ2, k, λ, tspan, tN)
    N = length(u0)
    Tmin, Tmax = tspan

    # Base linear operator X
    X = -ρ * L * (I + ϵ1 * M1 + ϵ2 * M2) - λ * I

    # Constant forcing term b = k * ones
    b = k .* ones(N)

    # Build augmented matrix
    Xtilde = zeros(N+1, N+1)
    Xtilde[1:N, 1:N] = X
    Xtilde[1:N, N+1] = b
    # last row remains zeros (dummy variable)

    # Augmented initial condition
    utilde0 = vcat(u0, 1.0)

    # create matrix output of simulation, rows = state, columns = time
    #Ts = range(Tmin, Tmax; length = tN)
    #U = zeros(length(utilde0), tN)
    #Δt = step(Ts)
    #expX = exp(Xtilde * Δt)

    #for (j, T) in enumerate(Ts)
    #    U[:, j] = exp(Xtilde * T) * utilde0
    #end

    # MORE EFFICIENT
    # create matrix output of simulation, rows = state, columns = time
    Ts = range(Tmin, Tmax; length = tN)
    U = zeros(length(utilde0), tN)
    Δt = step(Ts)

    expX = exp(Xtilde * Δt)               # transition matrix for one time step
    u = exp(Xtilde * Tmin) * utilde0      # correct initial state

    U[:,1] = u
    for j in 2:tN
        u = expX * u
        U[:,j] = u
    end

    # Return only physical variables (discard dummy)
    return U[1:N,:]
end


function disease_initiation_matrix(L, M1_matrix, M2_matrix, u0, ρ, ϵ1, ϵ2, k, λ, T)
    N = size(M1_matrix,2)
    S = size(M1_matrix,1)
    sol_matrix = zeros(S,N)
    for i in axes(M1_matrix,1)
        M1 = diagm(M1_matrix[i,:])
        M2 = diagm(M2_matrix[i,:])
        sol_matrix[i,:] = disease_initiation_vector(L, M1, M2, u0, ρ, ϵ1, ϵ2, k, λ, T)
    end
    return sol_matrix
end

# OLD MATRIX VERSION
#function epicenter_accuracy(pred_matrix, tau_matrix, k)
#    hits = zeros(size(pred_matrix,1))
#    for i in axes(pred_matrix,1)
#        epi_pred = sortperm(pred_matrix[i,:], rev=true)[1:k]
#        highest_tau_inds = sortperm(tau_matrix[i,:], rev=true)[1:k]
#        hits[i] = length(intersect(epi_pred, highest_tau_inds))/k  
#    end
#    return hits
#end
# NEW VECTOR VERSION
function epicenter_accuracy(prediction, observation, k)
    epi_pred = sortperm(prediction, rev=true)[1:k]
    obs_inds = sortperm(observation, rev=true)[1:k]
    hit =  length(intersect(epi_pred, obs_inds))/k  
    return hit
end



function load_dataset(name::Symbol; DX=nothing, centiloid_threshold=nothing)
    if name == :baseline_amyloid_tau
        return process_ADNI_A4_baseline(DX=DX)
    elseif name == :FDG_amyloid_tau_longitudinal
        return process_ADNI_HABS_FDG_amyloid_tau_longitudinal(DX=DX, centiloid_threshold=centiloid_threshold)
    else
        error("Unknown dataset: $name")
    end
end

function process_ADNI_A4_baseline(;DX=nothing)
    # read amyloid data from CSV
    df = CSV.read("data/ADNI_A4_cross-sectional_baseline_amy_tau.csv", DataFrame)
    if DX !== nothing
        df = df[df.DX .== DX, :]
    end
    amyloid_cols = filter(name -> startswith(name, "centiloid.amyloid.SUVR.Schaefer200"), names(df))
    amyloid_matrix = Matrix(df[:, amyloid_cols])

    # read tau data from CSV
    tau_cols = filter(name -> startswith(name, "tau.SUVR.Schaefer200"), names(df))
    tau_matrix = Matrix(df[:, tau_cols])
    return nothing, amyloid_matrix, tau_matrix
end

function process_ADNI_HABS_FDG_amyloid_tau_longitudinal(;DX=nothing, centiloid_threshold=nothing)
    df = CSV.read("data/ADNI_HABS_amyloid_FDG_longitudinal_tau.csv", DataFrame)
    if DX !== nothing
        # 1. Identify subjects whose FIRST scan has DX == target DX
        first_scan_mask = combine(groupby(df, :ID)) do subdf
            (; ID = subdf.ID[1], keep = (subdf.DX[1] == DX))
        end
    
        # 2. Extract the RIDs to keep
        good_RIDs = first_scan_mask.ID[first_scan_mask.keep .== true]
    
        # 3. Filter original df to keep *all rows* for these RIDs
        df = df[in.(df.ID, Ref(good_RIDs)), :]
    end

    for c in names(df)
        col = df[!, c]
        if eltype(col) <: AbstractString   # Pooled or not
            # allow missing values in this column
            df[!, c] = allowmissing(col)
            # now we can safely replace "NA"
            replace!(df[!, c], "NA" => missing)
        end
    end

    # Convert centiloid column to Float64, treating missing as +Inf so they don't pass the threshold
    if "centiloid" in names(df)
        df.centiloid = map(x -> x === missing ? Inf : parse(Float64, x), df.centiloid)
    end
    if centiloid_threshold !== nothing
        # 1. Identify subjects whose FIRST scan has centiloid < threshold
        first_scan_mask = combine(groupby(df, :ID)) do subdf
            (; ID = subdf.ID[1], keep = (subdf.centiloid[1] < centiloid_threshold))
        end
    
        # 2. Extract subject IDs to keep
        good_RIDs = first_scan_mask.ID[first_scan_mask.keep]
    
        # 3. Filter original df to keep *all scans* for these subjects
        df = df[in.(df.ID, Ref(good_RIDs)), :]
    end
    
    FDG_cols     = filter(c -> startswith(c, "FDG.SUVR.Schaefer200"), names(df))
    amyloid_cols = filter(c -> startswith(c, "centiloid.amyloid.SUVR.Schaefer200"), names(df))
    tau_cols     = filter(c -> startswith(c, "tau.SUVR.Schaefer200"), names(df))
    # Convert ROI columns to Float64 (or missing)
    for c in vcat(FDG_cols, amyloid_cols, tau_cols)
        df[!, c] = passmissing(x -> parse(Float64, x)).(df[!, c])
    end

    @assert "ID" in names(df)
    @assert "date" in names(df)

    # Convert date strings to Dates
    if df.date isa Vector{String}
        df.date = Date.(df.date, dateformat"yyyy-mm-dd")
    end

    groups = groupby(df, :ID)
    subjects = [g.ID[1] for g in groups]

    nsub = length(groups)
    nroi = length(FDG_cols)

    FDG_matrix     = Matrix{Union{Missing,Float64}}(missing, nsub, nroi)
    amyloid_matrix = Matrix{Union{Missing,Float64}}(missing, nsub, nroi)
    tau_matrix     = Matrix{Union{Missing,Float64}}(missing, nsub, nroi)


    for (i, g) in enumerate(groups)
        sort!(g, :date)

        FDG_matrix[i,:] = collect(g[1,FDG_cols])
        amyloid_matrix[i,:] = collect(g[1,amyloid_cols])
        tau_matrix[i,:] = collect(g[2,tau_cols])
    end

    return FDG_matrix, amyloid_matrix, tau_matrix
end

function drop_missing_rows(mats...)
    missing_rows = []
    # find nonmissing rows and add them to the collection of nonmissing rows
    for mat in mats
        if mat === nothing
            continue
        end
        idx = findall(i -> all(ismissing, mat[i, :]), 1:size(mat,1))
        append!(missing_rows, idx)
    end
    missing_rows = sort(unique(missing_rows))

    new_mats = Vector{Union{Nothing, Matrix{Float64}}}(undef,size(mats,1))
    for (i, mat) in enumerate(mats)
        if mat === nothing
            new_mats[i] = nothing
            continue
        end
        new_mats[i] = mat[sort(setdiff(1:size(mat,1), missing_rows)), :]  # drop missing rows
    end
    return new_mats
end


function spearman_with_pvalue(x, y)
    ρ = corspearman(x, y)
    n = length(x)
    # t-statistic
    t = ρ * sqrt((n - 2) / (1 - ρ^2))
    # two-sided p-value
    p = 2 * (1 - cdf(TDist(n - 2), abs(t)))
    return ρ, p
end

# ---------------------------------------------------------
# NORMALIZE ROWS OF PET MATRICES TO [0,1]
# ---------------------------------------------------------
function normalize_rows(M)
    if M === nothing
        return M
    end
    for i in axes(M,1)
        row = M[i, :]
        M[i, :] = (row .- minimum(row)) ./ (maximum(row) - minimum(row))
    end
    return M
end

# ================================
# HELPER: Time-sweep metric for one subject
# ================================
function subject_timesweep_metric(metric, init_timeseries, tau_row)
    Tn = size(init_timeseries, 2)
    vals = zeros(Tn)
    for j in 1:Tn
        pred = init_timeseries[:, j]
        targ = tau_row
        vals[j] = metric(pred, targ)
    end
    return maximum(vals[2:end])  # exclude t=0 (variance is zero there for simulations with homogeneous initial conditions)
end


# ================================
# HELPER: Build first objective (time sweep per subject)
# ================================
function make_objective_timesweep(metric, L, amyloid_matrix, FDG_matrix,
                                  tau_matrix, tspan, Tn)
    S = size(tau_matrix, 1)
    N = size(amyloid_matrix, 2)
    function objective(θ)
        ϵA, ϵF = θ
        scores = zeros(S)
        for i in 1:S
            init_timeseries = disease_initiation_timeseries(
                L,
                diagm(amyloid_matrix[i, :]),
                diagm(FDG_matrix[i, :]),
                ones(N),
                1, ϵA, ϵF, 0, 0,
                tspan,
                Tn
            )
            scores[i] = subject_timesweep_metric(
                metric, init_timeseries, tau_matrix[i, :]
            )
        end
        return -mean(scores)
    end
    return objective
end

function save_optimization_summary(
    filename::String;
    res,
    best_params,
    best_score,
    tspan,
    Tn
)
    open(filename, "w") do io
        println(io, "Optimization summary")
        println(io, "Timestamp: ", Dates.now())
        println(io, "Tspan = $(tspan)")
        println(io, "#timepoints = $(Tn)")

        println(io, "\nBest parameters:")
        for (i, p) in enumerate(best_params)
            println(io, "param[$i] = $(p)")
        end

        println(io, "\nBest score (mean metric): ", best_score)

        println(io, "\nBlackBoxOptim summary:\n")
        println(io, res)
    end
end


# ================================
# HELPER: Objective for single-time global model
# ================================
function make_objective_global(metric, L, amyloid_matrix, FDG_matrix,
                               tau_matrix, T)
    S = size(tau_matrix, 1)
    N = size(amyloid_matrix, 2)

    function objective(θ)
        ρ, ϵA, ϵF = θ

        init_matrix = disease_initiation_matrix(
            L,
            amyloid_matrix,
            FDG_matrix,
            ones(N),
            ρ, ϵA, ϵF, 0, 0,
            T
        )

        # SLOW VERSION
        #scores = [metric(init_matrix[i,:], tau_matrix[i,:]) for i in 1:S]   
        # FASTER
        scores = Vector{Float64}(undef, S)
        for i in 1:S
            @inbounds scores[i] = metric(view(init_matrix, i, :), view(tau_matrix, i, :))
        end
        return -mean(scores)
    end

    return objective
end


end
