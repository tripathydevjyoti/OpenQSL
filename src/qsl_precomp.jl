export
  QSL_OTOC,
  PrecompSpectralBound,
  PrecompStateBound,
  time_evol_state,
  interaction_picture,
  renyi_entropy,
  time_evol_jump,
  time_evol_jump_norm,
  limit_dm,
  number_quench,
  create_annihilation_creation_descending

###############################################################################
# 1. PRECOMPUTATION STRUCTS AND FUNCTIONS
###############################################################################

#############################
# For the State Bound
#############################


struct PrecompStateBound{T <: Real, I}
    t_vals::Vector{T}    # Time grid values
    g_vals::Vector{T}    # Precomputed inner integrals at grid points
    interpolation::I
end


function PrecompStateBound(t_max::T, num_points::Int, J::T, U::T,
                             full_ham, sys_ham, bath_ham, eigenvecs, 
                             rho::SparseMatrixCSC{ComplexF64,Int64},
                             is_thermal::Bool) where T <: Real
    # Create an evenly spaced grid for t₁
    t_vals = collect(range(zero(T), t_max, length=num_points))
    g_vals = similar(t_vals)
    # Define the state-bound integrand (same as in your compute_state_bound):
    function integrand_state(time1::T, time2::T)
        rho_t = time_evol_state(rho, full_ham, time1)
        rho_int_t = interaction_picture(full_ham.basis, U, rho_t)
        thermal_dm = thermal_state(1.0, full_ham.basis.N, full_ham.basis.M, J, U)
        rhob = is_thermal ? thermal_dm :
                partial_trace_system(rho_int_t, size(rho,1), full_ham.basis.N, full_ham.basis.M)
        bath = two_time_corr(bath_ham, eigenvecs, [time1, time2], rhob)
        a, adag = create_annihilation_creation_descending(full_ham.basis.N)
        norm1 = time_evol_jump_norm(time1, adag, sys_ham) * time_evol_jump_norm(time2, a, sys_ham)
        norm2 = time_evol_jump_norm(time1, a, sys_ham) * time_evol_jump_norm(time2, adag, sys_ham)
        term1 = (bath[1] + conj(bath[2])) * 2 * norm2
        term2 = (conj(bath[1]) + bath[2]) * 2 * norm1
        return J * J * (term1 + term2)
    end
    # Compute the inner integral g_state(t1) = ∫₀^(t1) integrand_state(t1,t2) dt2 at each grid point.
    for (i, t1) in enumerate(t_vals)
        g_vals[i] = real(quadgk(t2 -> integrand_state(t1, t2), zero(T), t1)[1])
    end
    interp = LinearInterpolation(t_vals, g_vals, extrapolation_bc=Line())
    return PrecompStateBound{T,typeof(interp)}(t_vals, g_vals, interp)
end


function compute_state_bound_precomp(t::T, psb::PrecompStateBound{T}) where T
    result, _ = quadgk(psb.interpolation, zero(T), t)
    return result
end

#############################
# For the Spectral Bound
#############################


struct PrecompSpectralBound{T <: Real, I}
    t_vals::Vector{T}    # Time grid values
    g_vals::Vector{T}    # Precomputed inner integrals at grid points
    interpolation::I
end


function PrecompSpectralBound(t_max::T, num_points::Int, J::T, U::T,
                              full_ham, sys_ham, bath_ham, eigenvecs,
                              rho::SparseMatrixCSC{ComplexF64,Int64},
                              is_thermal::Bool) where T <: Real
    t_vals = collect(range(zero(T), t_max, length=num_points))
    g_vals = similar(t_vals)
    # Define the liouvillian superoperator integrand as in compute_spectral_bound:
    function liouv_superop(time1::T, time2::T)
        a, adag = create_annihilation_creation_descending(full_ham.basis.N)
        a_t1 = time_evol_jump(time1, a, sys_ham)
        a_t2 = time_evol_jump(time2, a, sys_ham)
        adag_t1 = time_evol_jump(time1, adag, sys_ham)
        adag_t2 = time_evol_jump(time2, adag, sys_ham)
        rho_t = time_evol_state(rho, full_ham, time1)
        rho_int_t = interaction_picture(full_ham.basis, U, rho_t)
        thermal_dm = thermal_state(1.0, full_ham.basis.N, full_ham.basis.M, J, U)
        rhob = is_thermal ? thermal_dm :
                partial_trace_system(rho_int_t, size(rho,1), full_ham.basis.N, full_ham.basis.M)
        bath_corr = two_time_corr(bath_ham, eigenvecs, [time1, time2], rhob)
        iden = zeros(full_ham.basis.N + 1, full_ham.basis.N + 1)
        superop1 = (J * J) * bath_corr[1] * (kron(adag_t2, a_t1) - kron(iden, adag_t2 * a_t1))
        superop2 = (J * J) * conj(bath_corr[1]) * (kron(adag_t1, a_t2) - kron(adag_t1 * a_t2, iden))
        superop3 = (J * J) * bath_corr[2] * (kron(a_t2, adag_t1) - kron(iden, a_t2 * adag_t1))
        superop4 = (J * J) * conj(bath_corr[2]) * (kron(a_t1, adag_t2) - kron(a_t1 * adag_t2, iden))
        return superop1 + superop2 + superop3 + superop4 - adjoint(superop1 + superop2 + superop3 + superop4)
    end
    # For each t₁, compute the inner integral and then take the ∞–norm.
    for (i, t1) in enumerate(t_vals)
        # Integrate liouv_superop over t₂ from 0 to t1.
        op = real(quadgk(t2 -> liouv_superop(t1, t2), zero(T), t1)[1])
        g_vals[i] = norm(op, Inf)
    end
    interp = LinearInterpolation(t_vals, g_vals, extrapolation_bc=Line())
    return PrecompSpectralBound{T, typeof(interp)}(t_vals, g_vals, interp)
end


function compute_spectral_bound_precomp(t::T, pspecb::PrecompSpectralBound{T}) where T
    result, _ = quadgk(pspecb.interpolation, zero(T), t)
    return result
end


###############################################################################
# 2. MODIFIED QSL_OTOC STRUCT WITH OPTION TO USE PRECOMPUTATION
###############################################################################


struct QSL_OTOC{T <: Real}
    state_bound::T
    spectral_bound::T
end

# The “direct” inner integration functions (unchanged):
function compute_state_bound(t::T, J::T, U::T, full_ham, sys_ham, bath_ham, eigenvecs, 
                             rho::SparseMatrixCSC{ComplexF64,Int64}, is_thermal::Bool) where T <: Real
    function integrand(time1::T, time2::T)
        rho_t = time_evol_state(rho, full_ham, time1)
        rho_int_t = interaction_picture(full_ham.basis, U, rho_t)
        thermal_dm = thermal_state(1.0, full_ham.basis.N, full_ham.basis.M, J, U)
        rhob = is_thermal ? thermal_dm :
                partial_trace_system(rho_int_t, size(rho,1), full_ham.basis.N, full_ham.basis.M)
        bath = two_time_corr(bath_ham, eigenvecs, [time1, time2], rhob)
        a, adag = create_annihilation_creation_descending(full_ham.basis.N)
        norm1 = time_evol_jump_norm(time1, adag, sys_ham) * time_evol_jump_norm(time2, a, sys_ham)
        norm2 = time_evol_jump_norm(time1, a, sys_ham) * time_evol_jump_norm(time2, adag, sys_ham)
        term1 = (bath[1] + conj(bath[2])) * 2 * norm2
        term2 = (conj(bath[1]) + bath[2]) * 2 * norm1
        return J * J * (term1 + term2)
    end
    inner_integral(time1::T) = quadgk(time2 -> integrand(time1, time2), zero(T), time1)[1]
    return quadgk(inner_integral, zero(T), t)[1]
end

function compute_spectral_bound(t::T, J::T, U::T, full_ham, sys_ham, bath_ham, eigenvecs, 
                                rho::SparseMatrixCSC{ComplexF64,Int64}, is_thermal::Bool) where T <: Real
    function liouv_superop(time1::T, time2::T)
        a, adag = create_annihilation_creation_descending(full_ham.basis.N)
        a_t1 = time_evol_jump(time1, a, sys_ham)
        a_t2 = time_evol_jump(time2, a, sys_ham)
        adag_t1 = time_evol_jump(time1, adag, sys_ham)
        adag_t2 = time_evol_jump(time2, adag, sys_ham)
        rho_t = time_evol_state(rho, full_ham, time1)
        rho_int_t = interaction_picture(full_ham.basis, U, rho_t)
        thermal_dm = thermal_state(1.0, full_ham.basis.N, full_ham.basis.M, J, U)
        rhob = is_thermal ? thermal_dm :
                partial_trace_system(rho_int_t, size(rho,1), full_ham.basis.N, full_ham.basis.M)
        bath_corr = two_time_corr(bath_ham, eigenvecs, [time1, time2], rhob)
        iden = zeros(full_ham.basis.N + 1, full_ham.basis.N + 1)
        superop1 = (J * J) * bath_corr[1] * (kron(adag_t2, a_t1) - kron(iden, adag_t2 * a_t1))
        superop2 = (J * J) * conj(bath_corr[1]) * (kron(adag_t1, a_t2) - kron(adag_t1 * a_t2, iden))
        superop3 = (J * J) * bath_corr[2] * (kron(a_t2, adag_t1) - kron(iden, a_t2 * adag_t1))
        superop4 = (J * J) * conj(bath_corr[2]) * (kron(a_t1, adag_t2) - kron(a_t1 * adag_t2, iden))
        return superop1 + superop2 + superop3 + superop4 - adjoint(superop1 + superop2 + superop3 + superop4)
    end
    function inner_integral_spectral(time1::T)
        return norm(quadgk(time2 -> liouv_superop(time1, time2), zero(T), time1)[1], Inf)
    end
    return quadgk(inner_integral_spectral, zero(T), t)[1]
end


function QSL_OTOC(t::T, J::T, U::T, full_ham, sys_ham, bath_ham, eigenvecs,
                  rho::SparseMatrixCSC{ComplexF64,Int64}, is_thermal::Bool;
                  precomp_state::Union{PrecompStateBound{T},Nothing} = nothing,
                  precomp_spec::Union{PrecompSpectralBound{T},Nothing} = nothing) where T <: Real
    if precomp_state !== nothing && precomp_spec !== nothing
        sb = compute_state_bound_precomp(t, precomp_state)
        spb = compute_spectral_bound_precomp(t, precomp_spec)
    else
        sb = compute_state_bound(t, J, U, full_ham, sys_ham, bath_ham, eigenvecs, rho, is_thermal)
        spb = compute_spectral_bound(t, J, U, full_ham, sys_ham, bath_ham, eigenvecs, rho, is_thermal)
    end
    return QSL_OTOC{T}(real(sb), real(spb))
end


function time_evol_state(rho::SparseMatrixCSC{ComplexF64, Int64}, bh::BoseHubbard{T}, time::T) where T<: Number
    τ = -1im*time
    U = exponential!(Matrix(τ*bh.H))
    U_dag = adjoint(U)
   
    
    U*rho*U_dag
end  

function interaction_picture(basis, U, rho)
    dims = basis.dim
    rho_int_pic = zeros(ComplexF64, dims, dims)
    for (i,_) in enumerate(1:dims)
        for (j,_) in enumerate(1:dims)
            n1 = basis.eig_vecs[i][1]
            n2 = basis.eig_vecs[j][1]
            rho_int_pic[i,j] = rho[i,j]*exp(1im*0.5*U*n1*(n1-1))*exp(-1im*0.5*U*n2*(n2-1))
        end
    end
    rho_int_pic
end


function renyi_entropy(rho)
    rho_sq = rho*rho
    tr_rho_sq = real(tr(rho_sq))
    
    return -log(tr_rho_sq)

end   

function time_evol_jump_norm(time::T,op, ham::RBoseHubbard{T}) where T<:Real
    τ =-1im*time
    prop = exponential!(Matrix(τ*ham.H))
    prop_dag = adjoint(prop)
    evol_op = prop_dag*op*prop
    norm(evol_op, Inf)
end  

function time_evol_jump(time::T,op, ham::RBoseHubbard{T}) where T<:Real
    τ =-1im*time
    prop = exponential!(Matrix(τ*ham.H))
    prop_dag = adjoint(prop)
    evol_op = prop_dag*op*prop
   
end  


function limit_dm(rho::SparseMatrixCSC{T, Int64}, N::Int, M::Int) where {T<:Number}
    spinfo = findnz(rho)

    dims = NBasis(N, M).dim
    limit_rho = zeros(ComplexF64, dims, dims)

    for (i, val) in enumerate(spinfo[3])
        try
            index1 = get_index(NBasis(N, M), tensor_basis(N, M).eig_vecs[spinfo[1][i]])
            index2 = get_index(NBasis(N, M), tensor_basis(N, M).eig_vecs[spinfo[2][i]])

            # Only update if indices are valid
            limit_rho[index1, index2] = ComplexF64(val)

        catch e
            if isa(e, ErrorException)
                # If index is not found, just skip this iteration
                continue
            else
                rethrow(e)  # In case of unexpected errors, rethrow them
            end
        end
    end

    return sparse(limit_rho)
end

"""
function create_annihilation_creation_descending(N::Int)
    # Initialize (N+1)x(N+1) matrices
    a = zeros(ComplexF64, N+1, N+1)  # Annihilation operator
    adag = zeros(ComplexF64, N+1, N+1)  # Creation operator

    # Populate the matrices
    for n in 1:N
        a[n+1, n] = sqrt(N - n + 1)  # a lowers |N-n+1⟩ to |N-n⟩
        adag[n, n+1] = sqrt(N - n + 1)  # a† raises |N-n⟩ to |N-n+1⟩
    end

    return a, adag
end
"""
function create_annihilation_creation_descending(N::Int)
    # Initialize (N+1)x(N+1) matrices
    a = zeros(ComplexF64, N+1, N+1)    # Annihilation operator
    adag = zeros(ComplexF64, N+1, N+1) # Creation operator

    # Populate the matrices
    for n in 1:N
        a[n, n+1] = sqrt(n)         # a lowers |n⟩ to |n-1⟩ (a[2,3] = sqrt(3))
        adag[n+1, n] = sqrt(n+1)     # a† raises |n⟩ to |n+1⟩ (adag[3,2] = sqrt(3))
    end

    return a, adag
end


function number_quench(N,M)
    size = NBasis(N,M).dim
    vecs = NBasis(N,M).eig_vecs
    quench = zeros(size,size)
    for (i, state) in enumerate(vecs)
        
        for (j,_) in enumerate(vecs)
            quench[i,j] = state[1]
        end    
    end
    trace = tr(quench)

    
    sparse(quench/trace)
end