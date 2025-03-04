export 
    QSL_OTOC_pre


struct QSL_OTOC_pre{T <: Real}
    state_bound::T
    spectral_bound::T

    function QSL_OTOC_pre(t::T, J::T, U::T, full_ham::BoseHubbard{T}, 
                      sys_ham::RBoseHubbard{T},
                      bath_ham::Vector{RBoseHubbard{T}},
                      eigenvecs::Matrix{T},
                      rho::SparseMatrixCSC{ComplexF64, Int64},
                      is_thermal::Bool) where T <: Real
        state_bound = real(compute_state_bound_pre(t, J, U, full_ham, sys_ham, bath_ham, eigenvecs, rho, is_thermal))
        spectral_bound = real(compute_spectral_bound_pre(t, J, U, full_ham, sys_ham, bath_ham, eigenvecs, rho, is_thermal))
        new{T}(state_bound, spectral_bound)
    end

    # Precomputed integration for the state bound
    function compute_state_bound_pre(t::T, J::T, U::T, full_ham::BoseHubbard{T},
        sys_ham::RBoseHubbard{T},
        bath_ham::Vector{RBoseHubbard{T}},
        eigenvecs::Matrix{T},
        rho::SparseMatrixCSC{ComplexF64,Int64},
        is_thermal::Bool) where T <: Real

        # Precompute thermal state once if needed.
        thermal_dm = is_thermal ? thermal_state(1.0, full_ham.basis.N, full_ham.basis.M, J, U) : nothing

        # Define the inner integrand as a function of v for fixed u.
        # Here, time1 = u and time2 = v*u.
        function inner_integrand(v, u)
            local time1 = u
            local time2 = v * u
            local rho_t = time_evol_state(rho, full_ham, time1)
            local rho_int_t = interaction_picture(full_ham.basis, U, rho_t)
            local rhob = is_thermal ? thermal_dm : partial_trace_system(rho_int_t, size(rho,1), full_ham.basis.N, full_ham.basis.M)
            local bath = two_time_corr(bath_ham, eigenvecs, [time1, time2], rhob)
            local a, adag = create_annihilation_creation_descending(full_ham.basis.N)
            local norm1 = time_evol_jump_norm(time1, adag, sys_ham) * time_evol_jump_norm(time2, a, sys_ham)
            local norm2 = time_evol_jump_norm(time1, a, sys_ham) * time_evol_jump_norm(time2, adag, sys_ham)
            local term1 = (bath[1] + conj(bath[2])) * 2 * norm2
            local term2 = (conj(bath[1]) + bath[2]) * 2 * norm1
            return J^2 * (term1 + term2)
        end

        # Define F(u) = u * ∫₀¹ inner_integrand(v,u) dv.
        function F(u)
            local inner_val = quadgk(v -> inner_integrand(v, u), 0.0, 1.0; rtol=1e-6)[1]
            return inner_val * u
        end

        # Precompute F(u) on a grid over u ∈ [0, t].
        num_u = 100  # Adjust grid density as needed.
        u_vals = collect(range(0.0, stop=t, length=num_u))
        F_vals = [F(u) for u in u_vals]
        
        local itp = LinearInterpolation(u_vals, F_vals, extrapolation_bc=Line())
        return quadgk(u -> itp(u), 0.0, t; rtol=1e-6)[1]
    end   

    # Precomputed integration for the spectral bound
    function compute_spectral_bound_pre(t::T, J::T, U::T, full_ham::BoseHubbard{T},
        sys_ham::RBoseHubbard{T},
        bath_ham::Vector{RBoseHubbard{T}},
        eigenvecs::Matrix{T},
        rho::SparseMatrixCSC{ComplexF64,Int64},
        is_thermal::Bool) where T <: Real

        # Precompute thermal state once if needed.
        thermal_dm = is_thermal ? thermal_state(1.0, full_ham.basis.N, full_ham.basis.M, J, U) : nothing

        # Define liouv_superop as a function of time1 and time2.
        function liouv_superop(time1::T, time2::T, J::T) where T <: Real
            local a, adag = create_annihilation_creation_descending(full_ham.basis.N)
            local a_t1 = time_evol_jump(time1, a, sys_ham)
            local a_t2 = time_evol_jump(time2, a, sys_ham)
            local adag_t1 = time_evol_jump(time1, adag, sys_ham)
            local adag_t2 = time_evol_jump(time2, adag, sys_ham)
            local rho_t = time_evol_state(rho, full_ham, time1)
            local rho_int_t = interaction_picture(full_ham.basis, U, rho_t)
            local rhob = is_thermal ? thermal_dm : partial_trace_system(rho_int_t, size(rho,1), full_ham.basis.N, full_ham.basis.M)
            local bath_corr = two_time_corr(bath_ham, eigenvecs, [time1, time2], rhob)
            local iden = zeros(full_ham.basis.N + 1, full_ham.basis.N + 1)
            local superop1 = (J^2) * (bath_corr[1]) * (kron(adag_t2, a_t1) - kron(iden, adag_t2 * a_t1))
            local superop2 = (J^2) * (conj(bath_corr[1])) * (kron(adag_t1, a_t2) - kron(adag_t1 * a_t2, iden))
            local superop3 = (J^2) * (bath_corr[2]) * (kron(a_t2, adag_t1) - kron(iden, a_t2 * adag_t1))
            local superop4 = (J^2) * (conj(bath_corr[2])) * (kron(a_t1, adag_t2) - kron(a_t1 * adag_t2, iden))
            return superop1 + superop2 + superop3 + superop4 - adjoint(superop1 + superop2 + superop3 + superop4)
        end

        # Define the inner integrand for the spectral bound as a function of v for fixed u.
        function inner_integrand_spec(v, u)
            local time1 = u
            local time2 = v * u
            return norm(liouv_superop(time1, time2, J), Inf)
        end

        # Define F_spec(u) = u * ∫₀¹ inner_integrand_spec(v,u) dv.
        function F_spec(u)
            local inner_val = quadgk(v -> inner_integrand_spec(v, u), 0.0, 1.0; rtol=1e-6)[1]
            return inner_val * u
        end

        # Precompute F_spec(u) on a grid over u ∈ [0, t].
        num_u = 100  # Adjust grid density as needed.
        u_vals = collect(range(0.0, stop=t, length=num_u))
        F_vals = [F_spec(u) for u in u_vals]
        
        local itp = LinearInterpolation(u_vals, F_vals, extrapolation_bc=Line())
        return quadgk(u -> itp(u), 0.0, t; rtol=1e-6)[1]
    end
end
