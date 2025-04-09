using QuadGK         # For numerical integration (quadgk)
using Interpolations # For interpolating precomputed data




###############################################################################
# 3. USAGE IN THE MAIN TIME-STEPPING LOOP
###############################################################################

# Suppose your parameters are defined as follows:
N = 4           # Number of sites
M = 4           # Number of bosons
J = 4.0
U = 20.0
beta = 1.0
t_stop = 0.20
num_points = 30
np = pyimport("numpy")
times = np.linspace(0, t_stop, num_points)

# Example: set up the initial density matrix (using your functions)
pure_system = zeros(ComplexF64, N+1, N+1)
pure_system[N, N] = 1.0
thermal_dm = thermal_state(beta, N, M, J, U)
result_dm = sparse(kron(pure_system, thermal_dm))
init_state = limit_dm(result_dm, N, M)

# Define the required Hamiltonians (assumed constructors for BoseHubbard and RBoseHubbard)
H = BoseHubbard(N, M, J, U, :OBC)         # Full Hamiltonian
sys_ham = RBoseHubbard(N, 2, 0.0, U)        # System Hamiltonian
bath_ham = RBoseHubbard.([N+1, N], M, J, U)  # Bath Hamiltonian (vector of two elements)
eigenvals, eigenvecs = eigen!(Matrix(bath_ham[2].H))  # Eigen-decomposition for the bath

# Choose a small Crank-Nicolson time step for state evolution
dt_cn = 0.001

# Precompute the inner–integral functions (do this once for the maximum time)
precomp_state = PrecompStateBound(t_stop, 200, J, U, H, sys_ham, bath_ham, eigenvecs, init_state, true)
precomp_spec  = PrecompSpectralBound(t_stop, 200, J, U, H, sys_ham, bath_ham, eigenvecs, init_state, true)

# Prepare arrays to store results
renyi_ent_list = []
bound_list_born = []
spec_bound_list_born = []

@showprogress for (i, t) in enumerate(times)
    # Evolve the state using Crank-Nicolson propagation:
    rho_t = time_evol_state_cn(init_state, H, t, dt_cn)
    # Transform into the interaction picture
    rho_int_t = interaction_picture(NBasis(N, M), U, rho_t)
    # Partial trace over the bath to obtain the system density matrix:
    rho_S = partial_trace_bath(rho_int_t, N, M)
    push!(renyi_ent_list, renyi_entropy(rho_S))
    
    # Compute QSL bounds using the precomputed inner integrals (incremental integration).
    qsl_born = QSL_OTOC(t, J, U, H, sys_ham, bath_ham, eigenvecs, init_state, true;
                         precomp_state = precomp_state, precomp_spec = precomp_spec)
    push!(bound_list_born, qsl_born.state_bound)
    push!(spec_bound_list_born, qsl_born.spectral_bound)
end

# (You can now use these lists for further analysis or plotting.)
