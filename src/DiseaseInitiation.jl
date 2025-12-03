module DiseaseInitiation

export laplacian, disease_initiation_vector, disease_initiation_matrix, epicenter_accuracy, load_dataset, drop_missing_rows

using DifferentialEquations, LinearAlgebra, Dates, CSV, DataFrames


function make_disease_initiation(L, M)
    function f!(du, u, p, t)
        ρ, ϵ, k, λ = p
        du .= -ρ * L * (I + ϵ .* M) * u .+ k .- λ .* u
    end
    return f!
end

function laplacian(A; kind::Symbol = :out)
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
    return D - A
end
function disease_initiation_vector(L, M, u0, ρ, ϵ, k, λ, T)
    N = length(u0)

    # Base linear operator X
    X = -ρ * L * (I + ϵ*M) - λ * I

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


function disease_initiation_matrix(L, M_matrix, u0, ρ, ϵ, k, λ, T)
    N = size(M_matrix,2)
    S = size(M_matrix,1)
    sol_matrix = zeros(S,N)
    for i in axes(M_matrix,1)
        M = diagm(M_matrix[i,:])
        sol_matrix[i,:] = disease_initiation_vector(L, M, u0, ρ, ϵ, k, λ, T)
    end
    return sol_matrix
end


function epicenter_accuracy(pred_matrix, tau_matrix, k)
    hits = zeros(size(pred_matrix,1))
    for i in axes(pred_matrix,1)
        epi_pred = sortperm(pred_matrix[i,:], rev=true)[1:k]
        highest_tau_inds = sortperm(tau_matrix[i,:], rev=true)[1:k]
        hits[i] = length(intersect(epi_pred, highest_tau_inds))/k  
    end
    return hits
end




function load_dataset(name::Symbol; DX=nothing)
    if name == :baseline_amyloid_tau
        return process_ADNI_A4_baseline(DX=DX)
    elseif name == :FDG_amyloid_tau_longitudinal
        return process_ADNI_HABS_FDG_amyloid_tau_longitudinal(DX=DX)
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
    amyloid_matrix = amyloid_matrix ./ median(amyloid_matrix)  # normalize by median
    amyloid_matrix = clamp.(amyloid_matrix, 0, Inf)  # threshold at 0.0

    # read tau data from CSV
    tau_cols = filter(name -> startswith(name, "tau.SUVR.Schaefer200"), names(df))
    tau_matrix = Matrix(df[:, tau_cols])
    return nothing, amyloid_matrix, tau_matrix
end

function process_ADNI_HABS_FDG_amyloid_tau_longitudinal(;DX=nothing)
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

        #@info collect(g[1,FDG_cols])
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
        @info size(mat,1)
        idx = findall(i -> all(ismissing, mat[i, :]), 1:size(mat,1))
        append!(missing_rows, idx)
    end
    missing_rows = sort(unique(missing_rows))

    new_mats = Vector{Any}(undef,size(mats,1))
    for (i, mat) in enumerate(mats)
        if mat === nothing
            new_mats[i] = nothing
            continue
        end
        new_mats[i] = mat[sort(setdiff(1:size(mat,1), missing_rows)), :]  # drop missing rows
    end
    return new_mats
end

end