using Plots

# ---------------------------
# Define parameters and U/J range
# ---------------------------
N = 4                   # number of sites
M = 4                   # number of bosons
J = 4.0                 # hopping parameter
beta = 1.0              # inverse temperature
t_stop = 0.20           # maximum evolution time
num_points = 30         # number of time points
dt_cn = 0.001           # time step for Crank-Nicolson evolution
np = pyimport("numpy")
times = np.linspace(0, t_stop, num_points)
pure_system = zeros(ComplexF64, N+1, N+1)
pure_system[N, N] = 1.0

# Define the range of U/J values (10 values from 1.0 to 10.0)
UJ_values = range(1.0, stop=10.0, length=10)

# ---------------------------
# Loop over U/J values and compute QSL quantities
# ---------------------------
qsl_data = []  # container to store data for each U/J

for UJ in UJ_values
    U_current = UJ * J
    println("Processing U/J = $UJ (U = $U_current)")
    
    # Recompute the thermal density matrix and initial state using updated U
    thermal_dm_current = thermal_state(beta, N, M, J, U_current)
    result_dm_current = sparse(kron(pure_system, thermal_dm_current))
    init_state_current = limit_dm(result_dm_current, N, M)
    
    # Recompute Hamiltonians using the new U value
    H_current = BoseHubbard(N, M, J, U_current, :OBC)
    sys_ham_current = RBoseHubbard(N, 2, 0.0, U_current)
    bath_ham_current = RBoseHubbard.([N+1, N], M, J, U_current)
    eigenvals_current, eigenvecs_current = eigen!(Matrix(bath_ham_current[2].H))
    
    # Arrays to store the QSL quantities over time for this U/J value
    renyi_ent_list_current = Float64[]
    spec_bound_list_born_current = Float64[]
    
    for t in times
        # Evolve the state using Crank–Nicolson:
        rho_t = time_evol_state_cn(init_state_current, H_current, t, dt_cn)
        # Transform to the interaction picture (using U_current)
        rho_int_t = interaction_picture(NBasis(N, M), U_current, rho_t)
        # Partial trace over the bath degrees of freedom:
        rho_S = partial_trace_bath(rho_int_t, N, M)
        push!(renyi_ent_list_current, renyi_entropy(rho_S))
        
        # Compute the QSL bound using the Born approximation:
        qsl_born_current = QSL_OTOC_trapz(t, J, U_current, H_current, sys_ham_current,
                                          bath_ham_current, eigenvecs_current, init_state_current, true)
        push!(spec_bound_list_born_current, real(qsl_born_current.spectral_bound))
    end
    
    # Compute QSL values:
    # OTOC from the Renyi entropy: exp(–Renyi entropy)
    # Liouvillian–space bound from the spectral bound: exp(–spectral bound)
    otoc_vals = exp.(-renyi_ent_list_current)
    liouv_vals = exp.(-spec_bound_list_born_current)
    
    # Save the data for this U/J value as a NamedTuple:
    push!(qsl_data, (UJ=UJ, times=times, otoc=otoc_vals, liouv=liouv_vals))
end

# ---------------------------
# Create subplots for each U/J ratio (5 rows x 2 columns)
# ---------------------------
n_plots = length(qsl_data)
n_rows = 5
n_cols = 2

# Collect individual subplots in an array
subplots = []

for data in qsl_data
    # Set the subplot title using the "title" keyword in the plot call:
    p = plot(data.times, data.otoc, label="OTOC", lw=2,
             title="U/J = $(round(data.UJ, digits=2))", xlabel="Time", ylabel="QSL")
    plot!(p, data.times, data.liouv, label="Liouv", lw=2, ls=:dash)
    push!(subplots, p)
end

# Combine the subplots into a single plot grid without a global title
combined_plot = plot(subplots..., layout=(n_rows, n_cols), size=(1200, 800))
display(combined_plot)
savefig(combined_plot, "qsl_subplots_UJ.pdf")

# ---------------------------
# Compute and Plot Integrated Negative Deviation for each U/J
# ---------------------------
# For each U/J ratio, we compute the time–integrated negative deviation of
# (OTOC - Liouv) (i.e. any positive difference is set to zero before integration)
integrated_diffs = []

for data in qsl_data
    diff_vals = data.otoc .- data.liouv
    diff_filtered = map(x -> x > 0 ? 0.0 : x, diff_vals)
    dt_val = times[2] - times[1]
    integrated = sum(diff_filtered) * dt_val
    push!(integrated_diffs, (UJ=data.UJ, integrated=integrated))
end

p2 = plot(title="Integrated Negative Deviation vs U/J",
          xlabel="U/J", ylabel="Integrated Negative Deviation", marker=:circle)
plot!(p2, [d.UJ for d in integrated_diffs],
         [d.integrated for d in integrated_diffs],
         label="Integrated (OTOC - Liouv)")
display(p2)
savefig(p2, "integrated_negative_deviation_UJ.pdf")
