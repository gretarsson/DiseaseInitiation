using DifferentialEquations, LinearAlgebra
using Random, Plots
using DiseaseInitiation: laplacian, make_disease_initiation, disease_initiation_eqn
Random.seed!(1234)
N = 200
W = rand(N,N)
m = rand(N)

# random Laplacian and activity matrix
L = laplacian(W, kind=:out)
M = diagm(m)

# ODE
disease_initiation = make_disease_initiation(L, M)
p = (1,1,1,1)
prob = ODEProblem(disease_initiation, ones(N)  , (0.0, 10.0), p)
sol = solve(prob, Tsit5())
plot(sol)

# equation
sol_eqn = disease_initiation_eqn(L, M, ones(N), 1, 1, 1, 1, 10.0)


maximum(abs.(sol_eqn - sol(10.0)))  # should be very small
