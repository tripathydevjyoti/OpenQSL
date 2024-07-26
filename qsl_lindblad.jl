include("../src/QSL_Bose_Hubbard.jl")
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
using Trapz



function thermal_corr(
    beta::T, H::Vector{BoseHubbard{S}}, time::T, partition::T;  kwargs=()
    ) where{S, T <:Real}
    
    trsum1 = 0
    trsum2 = 0
    
   
    for (i, energy) in enumerate(eigenvals)
        
        gibbs = exp(-beta*energy)
        
        gamma1 = gibbs*bath(time, time, H, 1, 1, State(eigenvecs[:, i], H[2].basis) )
        trsum1 = trsum1 + gamma1 

        gamma2 = gibbs*bath2(time, time, H, 1, 1, State(eigenvecs[:, i], H[2].basis) )
        trsum2 = trsum2 + gamma2
       
    end
    
    corr_arr = [trsum1/ partition, trsum2/partition ] 
return corr_arr
end


function integrate_thermal_corr(beta, H, time_range)
    
    
    integral_trsum1 = trapz(time_range, trsum1_vals)
    #print("int1done")
    integral_trsum2 = trapz(time_range, trsum2_vals)
    
    return integral_trsum1, integral_trsum2
end


J = 4.0
U = 16.0
N=M=6             #no of sites and bosons in the chain
beta = 1.0        #inverse temperature
H = BoseHubbard.([N+1, N, N-1,N-2], M, J, U, :OBC) #BH hamiltonian 
eigenvals, eigenvecs = eigen(Matrix(H[2].H))
partition_function = part_func(1.0, H)  #partition function for the BH hamiltonian


np = pyimport("numpy")
time_range = np.linspace(0,0.9,100)
trsum1_vals = Float64[]
trsum2_vals = Float64[]

    
@showprogress for t in time_range
        corr_arr = thermal_corr(beta, H, t,partition_function)
        push!(trsum1_vals, np.real(corr_arr[1]))
        push!(trsum2_vals, np.real(corr_arr[2]))
        #print(t)
end

integral_trsum1, integral_trsum2 = integrate_thermal_corr(beta, H, time_range)


n=1
bound_sum = 0
for j in 0:n-1
    
    bound_sum = bound_sum + (j+1)*(np.real((integral_trsum1) )+ np.real( (integral_trsum2)) )

end



times = np.linspace(0,0.1,20)
plot(J*times, 4*J*J*bound_sum*times)
np.save("qsl_lind_bound.npy",  4*J*J*bound_sum*times)



