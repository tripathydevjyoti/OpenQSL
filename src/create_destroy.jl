export
    operate,
    create, destroy,
    creation, annihilation

"""
$(TYPEDSIGNATURES)
"""
function operate(ket::Vector{Int}, i::Int, c::Int)
    nket = copy(ket)
    nket[i] += c
    nket
end

create(ket::Vector{Int}, i::Int) = operate(ket, i, 1)
destroy(ket::Vector{Int}, i::Int) = operate(ket, i, -1)

function destroy(state::State{T}, i::Int, H::Vector{RBoseHubbard{Float64}}) where T
    n = length(H[2].basis.eig_vecs)
    L = length(H[2].basis.eig_vecs[1])
    
    # Check if state.eig_vecs is empty.
    if isempty(state.eig_vecs)
        # Define a default zero ket.
        zero_ket = zeros(Int, L)
        return State([0], [zero_ket])
    end

    vecs = Vector{Vector{Int}}(undef, length(state.eig_vecs))
    Threads.@threads for k in 1:length(state.eig_vecs)
        ket = state.eig_vecs[k]
        # Only attempt to operate if the ket is long enough and has a positive entry.
        if length(ket) >= i && ket[i] > 0
            vecs[k] = operate(ket, i, -1)
        else
            # If not, use a zero vector of length L.
            vecs[k] = zeros(Int, L)
        end
    end

    K = findall(x -> !all(iszero, x), vecs)
    if isempty(K)
        zero_ket = zeros(Int, L)
        return State([0], [zero_ket])
    else
        kets = vecs[K]
        new_coeffs = state.coeff[K] .* sqrt.(getindex.(kets, i) .+ 1)
        return State(new_coeffs, kets)
    end
end



function create(state::State{T}, i::Int, H::Vector{RBoseHubbard{Float64}} ) where T
    n = length(H[2].basis.eig_vecs)
    # Create a vector for the new kets using a concrete element type.
   
    vecs = Vector(undef, n)
    
    Threads.@threads for k in 1:n
        vecs[k] = operate(H[2].basis.eig_vecs[k], i, +1)
    end

    nonzero_indices = findall(!iszero, state.coeff)
    valid_indices = isempty(nonzero_indices) ? collect(1:n) : nonzero_indices

    kets = vecs[valid_indices]
    
    base_coeffs = isempty(nonzero_indices) ? ones(eltype(state.coeff), n) : state.coeff[valid_indices]
    new_coeffs = base_coeffs .* sqrt.(getindex.(kets, i))
    
    return State(new_coeffs, kets)
end


"""
$(TYPEDSIGNATURES)
"""
function annihilation(::Type{T}, B::S, i::Int) where {T <: Number, S <: AbstractBasis}
    n = length(B.eig_vecs)
    I, J, V = Int[], Int[], T[]
    for (v, ket) ∈ enumerate(B.eig_vecs)
        if ket[i] > 0
            push!(J, v)
            push!(I, get_index(B, operate(ket, i, -1)))
            push!(V, sqrt(T(ket[i])))
        end
    end
    sparse(I, J, V, n, n)
end
annihilation(B::Basis, i::Int) = annihilation(Float64, B::Basis, i::Int)

"""
$(TYPEDSIGNATURES)
"""
creation(::Type{T}, B::Basis, i::Int) where {T} = transpose(annihilation(T, B, i))


"""
function create(state::State{T}, i::Int) where T
    n = length(state.eig_vecs)
    vec_type = typeof(operate(state.eig_vecs[1], i, +1))
    vecs = Vector{vec_type}(undef, n)
    Threads.@threads for k ∈ 1:n
        ket = state.eig_vecs[k]
        vecs[k] =  operate(ket, i, +1)
    end
    

    valid_indices = [k for k in 1:n if !iszero(vecs[k]) && length(vecs[k]) >= i]
    kets = vecs[valid_indices]
    coeffs = state.coeff[valid_indices] .* sqrt.(getindex.(kets, i))
    
    State(coeffs, kets)
end
"""