# Quantum Speed Limit (QSL) for the Bose-Hubbard Model

This repository contains Julia code to compute the Quantum Speed Limit (QSL) for the Bose-Hubbard model. The QSL is computed using out-of-time-ordered correlators (OTOCs) and Renyi entropy to analyze the scrambling behavior of bosonic systems. For detailed explanation of the source function [View the documentation](./github.pdf)


---

## Getting Started

### Prerequisites

Make sure you have the following Julia packages installed:

```julia
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
```

---

## Running the QSL Computation

### Parameter Definitions

The following parameters are set for the simulation:

```julia
N = 3  # number of sites in the chain
M = 3  # number of bosons in the chain
J = 4.0 # hopping parameter
U = 8.5 # on-site potential
T = eltype(J)  # set data type
beta = 1.0  # inverse temperature
t_stop = 0.2
num_points = 30
np = pyimport("numpy")
times = np.linspace(0, t_stop, num_points)
```

---

### Steps to Run the Simulation

#### 1. Set the System State

```julia
pure_system = zeros(T, N+1, N+1)
pure_system[N, N] = 1.0
```

#### 2. Compute the Thermal Density Matrix

```julia
thermal_dm = thermal_state(beta, N, M, J, U)
```

#### 3. Construct the Initial Density Matrix

```julia
result_dm = sparse(kron(pure_system, thermal_dm))
init_state = limit_dm(result_dm, N, M)
```

#### 4. Define Required Hamiltonians

```julia
H = BoseHubbard(N, M, J, U, :OBC)
sys_ham = RBoseHubbard(N, 2, 0.0, U)
bath_ham = RBoseHubbard.([N+1, N], M, J, U)
eigenvals, eigenvecs = eigen!(Matrix(bath_ham[2].H))
```

#### 5. Compute QSL Bounds

```julia
renyi_ent_list = []
bound_list = []
bound_list_born = []
spec_bound_list = []
spec_bound_list_born = []

@showprogress for (i, t) in enumerate(times)
    rho_t = time_evol_state(init_state, H, t)
    rho_int_t = interaction_picture(NBasis(N,M), U, rho_t)
    rho_S = partial_trace_bath(rho_int_t, N, M)
    push!(renyi_ent_list, renyi_entropy(rho_S))
    rho_B = partial_trace_system(rho_int_t, size(init_state,1),N,M)
    qsl = QSL_OTOC(t, J, N, sys_ham, bath_ham, eigenvecs, rho_B)
    qsl_born = QSL_OTOC(t, J, N, sys_ham, bath_ham, eigenvecs, thermal_dm)
    push!(bound_list, real(qsl.state_bound))
    push!(spec_bound_list, real(qsl.spectral_bound))
    push!(bound_list_born, real(qsl_born.state_bound))
    push!(spec_bound_list_born, real(qsl_born.spectral_bound))
end
```

#### 6. Plot the Results

```julia
plot(
    times,
    [exp.(-2 * real(bound_list)),
     exp.(-2 * real(bound_list_born)),
     exp.(-spec_bound_list),
     exp.(-spec_bound_list_born),
     exp.(-renyi_ent_list)],
    label=["State space" "State space + Born" "Liouv space" "Liouv space + Born" "OTOC"]
)
xlabel!("time")
ylabel!("QSL")
savefig("qsl_bh.pdf")
```

---

## Functions Overview

### `BoseHubbard(N, M, J, U, bndr)`
Defines the full Hamiltonian for the system.

### `RBoseHubbard(N, M, J, U)`
Creates block diagonal matrices to represent subspaces.

### `thermal_state(beta, N, M, J, U)`
Computes the thermal density matrix.

### `limit_dm(rho, N, M)`
Reduces the dimensionality of a density matrix.

### `partial_trace_bath(init_dm, N, M)`
Traces out the bath from the density matrix.

### `partial_trace_system(init_dm, subsys_size, N, M)`
Traces out the system and retains the bath.

### `interaction_picture(basis, U, rho)`
Converts the density matrix to the interaction picture.

### `renyi_entropy(rho)`
Computes the Renyi entropy of the density matrix.

### `QSL_OTOC(t, J, N, sys_ham, bath_ham, eigenvecs, rhob)`
Computes the Quantum Speed Limit using system-bath interactions.

### `time_evol_state(rho, bh, time)`
Evolves the density matrix over time.

### `create_annihilation_creation_descending(N)`
Constructs annihilation and creation operators for the system.

---

## License

This project is licensed under the MIT License.

