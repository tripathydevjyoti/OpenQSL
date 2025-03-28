export
  QSL_OTOC_trapz,
  QSL_OTOC_2trapz

struct QSL_OTOC_trapz{T <: Real}
    state_bound::T
    spectral_bound::T

    function QSL_OTOC_trapz(t::T, J::T, U::T,
                        full_ham::BoseHubbard{T},
                        sys_ham::RBoseHubbard{T},
                        bath_ham::Vector{RBoseHubbard{T}},
                        eigenvecs::Matrix{T},
                        rho::SparseMatrixCSC{ComplexF64, Int64},
                        is_thermal::Bool) where T <: Real
        state_bound = real(compute_state_bound_trapz(t, J, U, full_ham, sys_ham, bath_ham, eigenvecs, rho, is_thermal))
        spectral_bound = real(compute_spectral_bound_trapz(t, J, U, full_ham, sys_ham, bath_ham, eigenvecs, rho, is_thermal))
        new{T}(state_bound, spectral_bound)
    end

    function compute_state_bound_trapz(t::T, J::T, U::T,
                                 full_ham::BoseHubbard{T},
                                 sys_ham::RBoseHubbard{T},
                                 bath_ham::Vector{RBoseHubbard{T}},
                                 eigenvecs::Matrix{T},
                                 rho::SparseMatrixCSC{ComplexF64, Int64},
                                 is_thermal::Bool) where T <: Real
        function integrand(time1::T, time2::T, J::T) where T <: Real
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
            J * J * (term1 + term2)
        end

        n_points = 100  # adjust for accuracy/performance trade-off
        times = collect(range(0, stop=t, length=n_points))
        inner_vals = [quadgk(time2 -> integrand(time, time2, J), 0, time)[1] for time in times]
        return trapz(times, inner_vals)
    end

    function compute_spectral_bound_trapz(t::T, J::T, U::T,
                                    full_ham::BoseHubbard{T},
                                    sys_ham::RBoseHubbard{T},
                                    bath_ham::Vector{RBoseHubbard{T}},
                                    eigenvecs::Matrix{T},
                                    rho::SparseMatrixCSC{ComplexF64, Int64},
                                    is_thermal::Bool) where T <: Real
        function liouv_superop(time1::T, time2::T, J::T) where T <: Real
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
            superop1 = (J * J) * (bath_corr[1]) * (kron(adag_t2, a_t1) - kron(iden, adag_t2 * a_t1))
            superop2 = (J * J) * (conj(bath_corr[1])) * (kron(adag_t1, a_t2) - kron(adag_t1 * a_t2, iden))
            superop3 = (J * J) * (bath_corr[2]) * (kron(a_t2, adag_t1) - kron(iden, a_t2 * adag_t1))
            superop4 = (J * J) * (conj(bath_corr[2])) * (kron(a_t1, adag_t2) - kron(a_t1 * adag_t2, iden))
    
            superop = superop1 + superop2 + superop3 + superop4
            superop - adjoint(superop)
        end

        function inner_integral_spectral(time1::T) where T <: Real
            val = quadgk(time2 -> liouv_superop(time1, time2, J), 0, time1)[1]
            norm(val, Inf)
        end

        n_points = 100  # adjust as needed
        times = collect(range(0, stop=t, length=n_points))
        inner_vals = [inner_integral_spectral(time) for time in times]
        return trapz(times, inner_vals)
    end
end



struct QSL_OTOC_2trapz{T <: Real}
    state_bound::T
    spectral_bound::T

    function QSL_OTOC_2trapz(t::T, J::T, U::T,
                        full_ham::BoseHubbard{T},
                        sys_ham::RBoseHubbard{T},
                        bath_ham::Vector{RBoseHubbard{T}},
                        eigenvecs::Matrix{T},
                        rho::SparseMatrixCSC{ComplexF64, Int64},
                        is_thermal::Bool) where T <: Real
        state_bound = real(compute_state_bound_2trapz(t, J, U, full_ham, sys_ham, bath_ham, eigenvecs, rho, is_thermal))
        spectral_bound = real(compute_spectral_bound_2trapz(t, J, U, full_ham, sys_ham, bath_ham, eigenvecs, rho, is_thermal))
        new{T}(state_bound, spectral_bound)
    end

    # Use trapz for both inner and outer integration
    function compute_state_bound_2trapz(t::T, J::T, U::T,
                                 full_ham::BoseHubbard{T},
                                 sys_ham::RBoseHubbard{T},
                                 bath_ham::Vector{RBoseHubbard{T}},
                                 eigenvecs::Matrix{T},
                                 rho::SparseMatrixCSC{ComplexF64, Int64},
                                 is_thermal::Bool) where T <: Real

        # Inner integrand as a function of time1 and time2
        function integrand(time1::T, time2::T, J::T) where T <: Real
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
            J * J * (term1 + term2)
        end

        n_outer = 100  # Number of outer discretization points
        outer_times = collect(range(0, stop=t, length=n_outer))
        outer_integral_values = similar(outer_times)

        # For each outer time point, perform the inner integration using trapz
        for (i, time1) in enumerate(outer_times)
            n_inner = 100  # Number of inner discretization points
            # Avoid an empty range if time1 is 0
            inner_times = time1 == 0 ? [0.0] : collect(range(0, stop=time1, length=n_inner))
            inner_vals = [integrand(time1, time2, J) for time2 in inner_times]
            outer_integral_values[i] = trapz(inner_times, inner_vals)
        end

        return trapz(outer_times, outer_integral_values)
    end

    function compute_spectral_bound_2trapz(t::T, J::T, U::T,
                                    full_ham::BoseHubbard{T},
                                    sys_ham::RBoseHubbard{T},
                                    bath_ham::Vector{RBoseHubbard{T}},
                                    eigenvecs::Matrix{T},
                                    rho::SparseMatrixCSC{ComplexF64, Int64},
                                    is_thermal::Bool) where T <: Real

        function liouv_superop(time1::T, time2::T, J::T) where T <: Real
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
            superop1 = (J * J) * (bath_corr[1]) * (kron(adag_t2, a_t1) - kron(iden, adag_t2 * a_t1))
            superop2 = (J * J) * (conj(bath_corr[1])) * (kron(adag_t1, a_t2) - kron(adag_t1 * a_t2, iden))
            superop3 = (J * J) * (bath_corr[2]) * (kron(a_t2, adag_t1) - kron(iden, a_t2 * adag_t1))
            superop4 = (J * J) * (conj(bath_corr[2])) * (kron(a_t1, adag_t2) - kron(a_t1 * adag_t2, iden))
    
            superop = superop1 + superop2 + superop3 + superop4
            superop - adjoint(superop)
        end

        # Compute the inner integral norm using trapz for each outer time point
        function inner_integral_spectral(time1::T) where T <: Real
            n_inner = 100
            inner_times = time1 == 0 ? [0.0] : collect(range(0, stop=time1, length=n_inner))
            integrand_vals = [norm(liouv_superop(time1, time2, J), Inf) for time2 in inner_times]
            trapz(inner_times, integrand_vals)
        end

        n_outer = 100
        outer_times = collect(range(0, stop=t, length=n_outer))
        outer_vals = [inner_integral_spectral(time1) for time1 in outer_times]
        return trapz(outer_times, outer_vals)
    end
end

