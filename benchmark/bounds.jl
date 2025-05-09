include("../src/ME_Bose_Hubbard.jl")
using .ME_Bose_Hubbard

using Test
using LightGraphs
using LabelledGraphs
using QuadGK
using Plots
using PyCall
using DifferentialEquations
using KrylovKit
using LinearAlgebra
using Combinatorics
using Arpack
using ProgressMeter
using Printf
using SparseArrays
using BlockDiagonals
using ExponentialUtilities
using LogExpFunctions
using Trapz

"""
Set Parameters
"""
N = 4 # number of sites in the chain  (int values only)                                      
M = 4  # number of bosons in the chain (int values only)
J = 4.0 # hopping parameter (float values only)
U = 16.0 # on-site potential (float values only) 
T = eltype(J)  # set data-type for the rest of the code
beta = 1.0  # inverse temperature

t_stop = 0.20
num_points = 30
np = pyimport("numpy")
times = np.linspace(0, t_stop, num_points)

#red_ham = [RBoseHubbard(N+1,M,J,U), RBoseHubbard(N,M,J,U)] 

"""
Step 1: Set the state of the system to contain only 1 boson
"""
pure_system = zeros(ComplexF64, N+1, N+1)   # matrix of zeros for system dm that can contain from 0 to N bosons
pure_system[N, N] = 1.0                     # The first element is 1, rest are zeros

"""
Step 2: Get the thermal density matrix 
"""
thermal_dm = thermal_state(beta, N, M, J, U)    # creates a block diagonal thermal dm with N-1 bosons and M-1 sites
size(thermal_dm)

"""
Step 3: Take the tensor product of the density matrix and thermal density matrix
"""
result_dm = sparse(kron(pure_system, thermal_dm)) 

"""
Step 4: Limit the joint state to that of N bosons in total and apply the quench to set the initial state
(Quench process has been commented out for now.)
"""
#number_quench(N,M)
#number_quench_dag = SparseArrays.transpose(number_quench(N,M))
#result_dm
#limit_dm(result_dm,N,M)
#init_state = number_quench(N,M)*limit_dm(result_dm,N,M)*number_quench_dag

init_state = limit_dm(result_dm, N, M)

"""
Define the required Hamiltonians
"""
H = BoseHubbard(N, M, J, U, :OBC)     # Full Hamiltonian of N bosons and M sites
sys_ham = RBoseHubbard(N, 2, 0.0, U)    # System Hamiltonian of N bosons and 1 site
bath_ham = RBoseHubbard.([N+1, N], M, J, U)  # Bath Hamiltonian of N bosons and M-1 sites
eigenvals, eigenvecs = eigen!(Matrix(bath_ham[2].H))  # eigenvalues and eigenvectors of the bath hamiltonian used to trace over states

# =============================================================================
# Crank–Nicolson Time Evolution Functions
# =============================================================================

"""
    crank_nicolson_propagator(H::BoseHubbard{T}, dt::Real) where T<:Real

Compute the one–step Crank–Nicolson propagator for the Bose–Hubbard Hamiltonian.
Here H is of type BoseHubbard and H.H extracts its matrix.
Uses LU decomposition to solve
    (I + i*(dt/2)*H) * U = (I - i*(dt/2)*H)
"""
function crank_nicolson_propagator(H::BoseHubbard{T}, dt::Real) where T<:Real
    ham = H.H
    I_mat = Matrix{eltype(ham)}(I, size(ham, 1), size(ham, 1))
    A = I_mat - im*(dt/2)*ham
    B = I_mat + im*(dt/2)*ham
    # Use LU decomposition for solving B * U = A
    LU = lu(B)
    U = LU \ A
    return U
end

"""
    time_evol_state_cn(rho, H, t_total, dt_cn)

Evolves the state (or density matrix) `rho` under Hamiltonian `H` using the Crank–Nicolson 
method for a total time `t_total`. The evolution is split into small steps of duration `dt_cn`.
If t_total is not an integer multiple of dt_cn, dt_effective = t_total/n_steps is used.
"""
function time_evol_state_cn(rho::AbstractMatrix{ComplexF64}, H::BoseHubbard{T}, t_total::Real, dt_cn::Real) where T<:Real
    n_steps = max(1, Int(round(t_total / dt_cn)))
    dt_effective = t_total / n_steps
    U = crank_nicolson_propagator(H, dt_effective)
    ρ = rho
    for step in 1:n_steps
        ρ = U * ρ * U'
    end
    return ρ
end

# =============================================================================
# Main Loop for QSL and OTOC (modified to use Crank–Nicolson propagation)
# =============================================================================

# Define empty lists for the bounds and otoc/renyi entropy
renyi_ent_list = []
bound_list = []
bound_list_born = []
spec_bound_list = []
spec_bound_list_born = []

bath_ham[2].basis.eig_vecs

# Choose a small time step for the Crank–Nicolson scheme
dt_cn = 0.001

# Precompute the inner–integral functions (do this once for the maximum time)
precomp_state = PrecompStateBound(t_stop, 100, J, U, H, sys_ham, bath_ham, eigenvecs, init_state, true)
precomp_spec  = PrecompSpectralBound(t_stop, 100, J, U, H, sys_ham, bath_ham, eigenvecs, init_state, true)

# Initialize variables for incremental integration.
cumulative_state_bound = 0.0
cumulative_spec_bound  = 0.0
t_prev = 0.0  # starting at time zero

@showprogress for (i, t) in enumerate(times)


    global cumulative_state_bound, cumulative_spec_bound, t_prev

    # Evolve the state using Crank-Nicolson propagation:
    rho_t = time_evol_state_cn(init_state, H, t, dt_cn)
    # Transform into the interaction picture:
    rho_int_t = interaction_picture(NBasis(N, M), U, rho_t)
    # Partial trace over the bath to obtain the system density matrix:
    rho_S = partial_trace_bath(rho_int_t, N, M)
    push!(renyi_ent_list, renyi_entropy(rho_S))
    
    # Perform incremental integration between the previous time and current time t.
    # Integrate the precomputed state-bound inner integral:
    Δstate, _ = quadgk(precomp_state.interpolation, t_prev, t)
    cumulative_state_bound += Δstate
    
    # Integrate the precomputed spectral-bound inner integral:
    Δspec, _  = quadgk(precomp_spec.interpolation, t_prev, t)
    cumulative_spec_bound  += Δspec
    
    # Store the cumulative (outer) integration results at time t.
    push!(bound_list_born, cumulative_state_bound)
    push!(spec_bound_list_born, cumulative_spec_bound)
    
    t_prev = t  # Update the previous time for the next increment.
end


plot(
    times, 
    [exp.(-2 * real(bound_list_born)), 
     exp.(-spec_bound_list_born), 
     exp.(-renyi_ent_list)], 
    label = ["State space + Thermal bath" "Liouv space + Thermal bath" "OTOC"],
    lw = 2  # Increase line thickness
)
xlabel!("time")
ylabel!("QSL")
title!("Scrambling in BH (Crank-Nicolson)")
savefig("qsl_bh_N$(N)_J$(J)_U$(U)_highres.pdf")

np = pyimport("numpy")
np.save("renyi_ent_list.npy", renyi_ent_list)
np.save("bound_list_born.npy", bound_list_born)
np.save("spec_bound_list_born.npy", spec_bound_list_born)









