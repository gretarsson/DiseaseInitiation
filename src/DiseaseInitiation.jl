module DiseaseInitiation

export laplacian, disease_initiation_vector, disease_initiation_matrix, epicenter_accuracy, load_dataset, drop_missing_rows
export disease_initiation_timeseries
export spearman_with_pvalue, top1_rank_score, mean_epi_rank
export normalize_rows
export make_objective_timesweep, make_objective_global, save_local_timesweep_results
export save_optimization_summary
export zscore_PET, FDG_matrix_reference
export read_epicenter_subjects_and_intersections

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
    #S = max.(0.0, I + ϵ1 * M1 + ϵ2 * M2)  # enforce nonnegativity (no negative edges)
    #X = -ρ * L * S - λ * I

    # trying out scaling rows as well (need to change make_objective_timesweep too)
    S1 = max.(0.0, I + ϵ1 * M1 + ϵ2 * M2)  # enforce nonnegativity (no negative edges)
    S2 = max.(0.0, I + k * M1 + λ * M2)
    k = 0; λ = 0  # just testing with row-scaling and cant be botherec tod efine new parameters
    scaled_L = S1*L*S2
    scaled_L[diagind(L)] .= vec(sum(scaled_L, dims=1))
    X = -ρ * scaled_L * S - λ * I

    #X = -ρ * L * (I + ϵ1 * M1 + ϵ2 * M2) - λ * I

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


    # trying out scaling rows as well (need to change make_objective_timesweep too)
    #S1 = max.(0.0, I + ϵ1 * M1 + ϵ2 * M2)  # enforce nonnegativity (no negative edges)
    #S2 = max.(0.0, I + k * M1 + λ * M2)
    #k = 0; λ = 0  # just testing with row-scaling and cant be botherec tod efine new parameters
    #scaled_L = S2*L*S1
    #scaled_L[diagind(L)] .= vec(sum(scaled_L, dims=1))
    #X = -ρ * scaled_L  - λ * I

    # try interaction term
    #X = -ρ * L * (I + ϵ1 * M1 + ϵ2 * M2 + k * M1 .* M2) - λ * I
    #k = 0

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

function epicenter_accuracy(prediction, observation, k)
    epi_pred = sortperm(prediction, rev=true)[1:k]
    obs_inds = sortperm(observation, rev=true)[1:k]
    hit =  length(intersect(epi_pred, obs_inds))/k  
    return hit
end

"""
    mean_epi_rank_in_pred(pred, obs, epi_idx; descending=true)

Mean rank of experimental epicenter indices `epi_idx` in the model
prediction vector `pred` (higher = more epicenter-like).

`obs` is accepted for interface compatibility but is not used.

Ranks are 1-based; rank 1 = highest pred if descending=true.

Returns: Float64
"""
function mean_epi_rank(
    pred::AbstractVector{<:Real},
    obs::AbstractVector{<:Real},      # unused, kept for interface compatibility
    epi_idx::AbstractVector{Int};
    descending::Bool = true,
)::Float64
    @assert length(pred) == length(obs) "pred and obs must have same length"
    @assert all(1 .<= epi_idx .<= length(pred)) "epi_idx out of bounds"

    order = sortperm(pred; rev=descending)

    # region -> rank
    rank = zeros(Int, length(pred))
    for (r, i) in enumerate(order)
        rank[i] = r
    end

    return -mean(rank[epi_idx])
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

    return FDG_matrix, amyloid_matrix, tau_matrix, subjects
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
    nonmissing_rows = nothing

    new_mats = Vector{Union{Nothing, Matrix{Float64}}}(undef,size(mats,1))
    for (i, mat) in enumerate(mats)
        if mat === nothing
            new_mats[i] = nothing
            continue
        end
        nonmissing_rows = sort(setdiff(1:size(mat,1), missing_rows))
        new_mats[i] = mat[nonmissing_rows, :]  # drop missing rows
    end
    return (new_mats..., nonmissing_rows)
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

"""
Normalize metric input so that each individual i has its own function metric[i].

Accepts:
- metric::Function                 -> replicated S times
- metric::AbstractVector{<:Function} -> must have length S
"""
function normalize_metric(metric, S::Int)
    if metric isa Function
        return fill(metric, S)
    elseif metric isa AbstractVector{<:Function}
        length(metric) == S ||
            throw(ArgumentError("metric has length $(length(metric)); expected S = $S"))
        return metric
    else
        throw(ArgumentError("metric must be a Function or Vector{<:Function}"))
    end
end



# ================================
# HELPER: Build first objective (time sweep per subject)
# ================================
function make_objective_timesweep(metric, L, amyloid_matrix, FDG_matrix,
                                  tau_matrix, tspan, Tn)
    S = size(tau_matrix, 1)
    N = size(amyloid_matrix, 2)
    metrics = normalize_metric(metric, S)
    function objective(θ)
        ϵA, ϵF = θ
        #ϵA, ϵF, k = θ
        #ϵA, ϵF, k, λ = θ  # trying out scaling in and out degree
        scores = zeros(S)
        for i in 1:S
            init_timeseries = disease_initiation_timeseries(
                L,
                diagm(amyloid_matrix[i, :]),
                diagm(FDG_matrix[i, :]),
                ones(N),
                1, ϵA, ϵF, 0, 0,
                #1, ϵA, ϵF, k, λ,  # trying out scaling in and out degree
                tspan,
                Tn
            )
            scores[i] = subject_timesweep_metric(
                metrics[i], init_timeseries, tau_matrix[i, :]
            )
        end
        return -median(scores)
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

"""
    top1_rank_score(pred::AbstractVector, obs::AbstractVector) -> Float64

Score in [1/N, 1], where N = length(pred).
1.0 if argmax(pred) is also argmax(obs), decreasing as its rank in obs worsens.
"""
function top1_rank_score(pred::AbstractVector, obs::AbstractVector)
    @assert length(pred) == length(obs)
    N = length(pred)

    # index of highest predicted value
    _, idx_pred = findmax(pred)

    # sort observed in descending order, get permutation
    order = sortperm(obs; rev = true)          # order[k] = index of k-th largest obs

    # inverse permutation: rank[idx] = rank (1 = best)
    ranks = invperm(order)

    rank_pred = ranks[idx_pred]                # 1..N
    return (N - rank_pred + 1) / N
end



function save_local_timesweep_results(subject_fits, file_csv::String)
    S = length(subject_fits)

    df = DataFrame(
        subject = [sf.ID for sf in subject_fits],
        epsilon_A = [sf.best_params[1] for sf in subject_fits],
        epsilon_F = [sf.best_params[2] for sf in subject_fits],
        best_score = [sf.best_score for sf in subject_fits],
        amyloid_score = [sf.amyloid_score for sf in subject_fits],
        FDG_score = [sf.FDG_score for sf in subject_fits],
    )

    CSV.write(file_csv, df)
    println("Saved per-subject local timesweep results to\n → $file_csv")

    return df, file_csv
end


"""
    zscore_FDG(FDG_matrix, FDG_ref_matrix; eps=1e-12)

Z-score FDG_matrix ROI-wise using mean/std computed from FDG_ref_matrix.

- Columns = ROIs
- Rows = subjects
- Missing values are ignored when computing reference statistics
- Missing values are preserved in the output
"""
function zscore_PET(
    FDG_matrix::AbstractMatrix{<:Union{Missing,Real}},
    FDG_ref_matrix::AbstractMatrix{<:Union{Missing,Real}};
    eps::Float64 = 1e-12
)
    S, N = size(FDG_matrix)
    @assert size(FDG_ref_matrix, 2) == N

    μ = zeros(Float64, N)
    σ = zeros(Float64, N)

    # Compute reference mean/std per ROI
    for j in 1:N
        ref_vals = Float64[]
        for i in axes(FDG_ref_matrix, 1)
            x = FDG_ref_matrix[i, j]
            if x !== missing
                push!(ref_vals, float(x))
            end
        end
        μ[j] = mean(ref_vals)
        σj = std(ref_vals)
        σ[j] = σj < eps ? 1.0 : σj
    end

    # Z-score target matrix
    Z = Matrix{Union{Missing,Float64}}(missing, S, N)
    for i in 1:S, j in 1:N
        x = FDG_matrix[i, j]
        Z[i, j] = x === missing ? missing : (float(x) - μ[j]) / σ[j]
    end

    return Z
end


# build FDG reference
function FDG_matrix_reference()

    csv_path = "data/ADNI_HABS_amyloid_FDG_longitudinal_tau.csv"
    df = CSV.read(csv_path, DataFrame)

    @assert "DX" in names(df) "Column DX not found in CSV."
    @assert "ID" in names(df) "Column ID not found in CSV."

    # -----------------------------
    # Clean up "NA" strings -> missing (only for string columns)
    # -----------------------------
    for c in names(df)
        col = df[!, c]
        if eltype(col) <: AbstractString
            df[!, c] = allowmissing(col)
            replace!(df[!, c], "NA" => missing)
        end
    end

    # -----------------------------
    # Filter CN rows (keep ALL visits that are CN)
    # -----------------------------
    df_cn = df[df.DX .== "CN", :]

    println("Total rows: $(nrow(df))")
    println("CN rows:    $(nrow(df_cn))")

    # -----------------------------
    # Identify FDG columns (match your existing convention)
    # -----------------------------
    FDG_cols = filter(c -> startswith(c, "FDG.SUVR.Schaefer200"), names(df_cn))
    @assert !isempty(FDG_cols) "No FDG columns found. Check column prefixes in the CSV."

    # -----------------------------
    # Parse FDG columns to Float64 (allow missing)
    # -----------------------------
    for c in FDG_cols
        # Works if the column is already numeric or a String/Union with missing
        df_cn[!, c] = passmissing(x -> x isa Real ? Float64(x) : parse(Float64, x)).(df_cn[!, c])
    end

    # -----------------------------
    # Build FDG_matrix (rows = CN visits, cols = ROIs)
    # -----------------------------
    FDG_matrix = Matrix(df_cn[:, FDG_cols])  # Matrix{Union{Missing,Float64}} likely

    # Optional: keep metadata aligned with rows of FDG_matrix
    IDs   = df_cn.ID
    dates = ("date" in names(df_cn)) ? df_cn.date : nothing
    if dates !== nothing && eltype(dates) <: AbstractString
        # adjust dateformat if yours differs
        df_cn.date = Date.(df_cn.date, dateformat"yyyy-mm-dd")
        dates = df_cn.date
    end
    nonmissing_rows = findall(i -> !all(ismissing, FDG_matrix[i, :]), 1:size(FDG_matrix, 1))
    FDG_matrix = FDG_matrix[nonmissing_rows, :]
    return Matrix{Float64}(FDG_matrix)
end


"""
Read subject IDs and ROI intersections from
`subjects_with_epicenters_ALL_metrics.txt`.

Returns
-------
subject_ids :: Vector{String}
    Subject IDs (one per subject)

roi_intersections :: Vector{Vector{Int}}
    Intersection of ROI indices for each subject
    (original indexing; may be empty)
"""
function read_epicenter_subjects_and_intersections(txt_path::AbstractString)
    subject_ids = String[]
    roi_intersections = Vector{Vector{Int}}()

    open(txt_path, "r") do io
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            startswith(line, "#") && continue

            # Expected format:
            # SubjectID<TAB>MAD_ROIs<TAB>IQR_ROIs<TAB>RANKGAP_ROIs<TAB>Intersection<TAB>Union
            fields = split(line, '\t')
            @assert length(fields) ≥ 5 "Malformed line:\n$line"

            push!(subject_ids, fields[1])

            # Parse intersection column (field 5)
            if isempty(fields[5])
                push!(roi_intersections, Int[])
            else
                rois = parse.(Int, split(fields[5], ','))
                push!(roi_intersections, rois)
            end
        end
    end

    return subject_ids, roi_intersections
end



end

