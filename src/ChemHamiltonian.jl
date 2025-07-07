# This file is for development purposes only and will be removed once this first version is stable.


using TensorKit
using HTTN
using LinearAlgebra
using SparseArrays
using DataStructures: DefaultDict

# Define a single-site operator
struct SiteOp
    coefficient::Number
    operator::String  # Operator type like "a↑", "c↓", etc.
    site::Int         # Site index
    
    # Constructor with coefficient
    SiteOp(coef::Number, op::String, site::Int) = new(coef, op, site)
    
    # Default constructor with coefficient = 1.0
    SiteOp(op::String, site::Int) = new(1.0, op, site)
end

# Define equality for SiteOp (ignoring coefficient, just comparing operator and site)
function Base.:(==)(a::SiteOp, b::SiteOp)
    return a.operator == b.operator && a.site == b.site
end

# Define hash for SiteOp (ignoring coefficient for hash as well)
function Base.hash(op::SiteOp, h::UInt)
    return hash(op.operator, hash(op.site, h))
end

# Define multiplication for SiteOp
function Base.:*(a::Number, op::SiteOp)
    # Returns a new SiteOp with multiplied coefficient
    return SiteOp(a * op.coefficient, op.operator, op.site)
end

# Commutative multiplication
function Base.:*(op::SiteOp, a::Number)
    return a * op
end

# Define zero for SiteOp to be able to display it
Base.zero(::Type{Vector{SiteOp}}) = Vector{SiteOp}()

# Define a term in the operator sum
struct OpTerm
    coefficient::Number
    operators::Vector{String}  # Operator types like "a↑", "c↓", etc.
    sites::Vector{Int}         # Site indices
    
    function OpTerm(coef::Number, ops_sites...)
        ops = String[]
        sites = Int[]
        
        for i in 1:2:length(ops_sites)
            push!(ops, ops_sites[i])
            push!(sites, ops_sites[i+1])
        end
        
        return new(coef, ops, sites)
    end

    function OpTerm(coefficient::Number, ops::Vector{String}, sites::Vector{Int})
        if length(ops) != length(sites)
            throw(ArgumentError("Operators and sites must have the same length"))
        end
        return new(coefficient, ops, sites)
    end
end

function Base.:*(term::OpTerm, scalar::Number)
    # Multiply the coefficient of the term by a scalar
    return OpTerm(term.coefficient * scalar, term.operators, term.sites)
end

function Base.:*(scalar::Number, term::OpTerm)
    # Commutative multiplication
    return term * scalar
end

function Base.show(io::IO, term::OpTerm)
    if isempty(term.operators)
        print(io, "OpTerm($(term.coefficient), [], [])")
        return
    end
    
    # Group operators by site
    site_ops = Dict{Int, Vector{String}}()
    for (op, site) in zip(term.operators, term.sites)
        if !haskey(site_ops, site)
            site_ops[site] = String[]
        end
        push!(site_ops[site], op)
    end
    
    # Sort sites for consistent ordering
    sorted_sites = sort(collect(keys(site_ops)))
    
    # Build the formatted string
    site_strings = String[]
    for site in sorted_sites
        ops_at_site = join(site_ops[site], " ")
        push!(site_strings, "[ $ops_at_site ]_$site")
    end
    
    # Join with arrows and add factor
    formatted_term = join(site_strings, " ── ")
    print(io, "$formatted_term    factor = $(term.coefficient)")
end

# OpSum structure to collect and manage operator terms
struct OpSum
    terms::Vector{OpTerm}
    
    OpSum() = new(OpTerm[])
    OpSum(terms::Vector{OpTerm}) = new(terms)
    OpSum(term::OpTerm) = new([term])
end

# Add a term to the OpSum
function Base.:+(opsum::OpSum, args)
    if length(args) % 2 == 1 && !isa(args[1], Number)
        throw(ArgumentError("Expected coefficient followed by operator/site pairs"))
    end
    
    coef = args[1]
    ops_sites = args[2:end]
    
    term = OpTerm(coef, ops_sites...)
    push!(opsum.terms, term)
    
    return opsum
end

function Base.:+(opsum::OpSum, term::OpTerm)
    # Add a single OpTerm to the OpSum
    push!(opsum.terms, term)
    return opsum
end

function Base.:+(opsum::OpSum, terms::Vector{OpTerm})
    # Add multiple OpTerms to the OpSum
    for term in terms
        push!(opsum.terms, term)
    end
    return opsum
end

function Base.:+(opsum1::OpSum, opsum2::OpSum)
    # Combine two OpSums
    new_terms = vcat(opsum1.terms, opsum2.terms)
    return OpSum(new_terms)
end

function Base.:*(opsum::OpSum, scalar::Number)
    # Multiply all terms in the OpSum by a scalar
    new_terms = [term * scalar for term in opsum.terms]
    return OpSum(new_terms)
end

Base.:*(scalar::Number, opsum::OpSum) = opsum * scalar

# Iterator interface for OpSum
Base.getindex(ops::OpSum, i::Int) = ops.terms[i]
Base.getindex(ops::OpSum, r::AbstractRange) = OpSum(ops.terms[r])
Base.getindex(ops::OpSum, indices) = OpSum(ops.terms[collect(indices)])
Base.iterate(ops::OpSum) = isempty(ops.terms) ? nothing : (ops.terms[1], 1)
Base.keys(ops::OpSum) = 1:length(ops.terms)
Base.iterate(ops::OpSum, state) = state >= length(ops.terms) ? nothing : (ops.terms[state+1], state+1)
Base.length(ops::OpSum) = length(ops.terms)
Base.eltype(::Type{OpSum}) = OpTerm

function Base.show(io::IO, ops::OpSum)
    if isempty(ops.terms)
        print(io, "OpSum: (empty)")
    else
        print(io, "OpSum with $(length(ops.terms)) terms:\n")
        for (i, term) in enumerate(ops.terms)
            print(io, "  $i. ")
            show(io, term)
            if i < length(ops.terms)
                print(io, "\n")
            end
        end
    end
end

# For MIME display in Jupyter (optional, for even better formatting)
function Base.show(io::IO, ::MIME"text/plain", ops::OpSum)
    show(io, ops)
end

"""
    VirtSpace

A struct representing a virtual space with spin-up and spin-down quantum numbers.
Used in MPO construction for quantum chemistry Hamiltonians.

Fields:
- qn1::Int: For non-spin symmetry, it represents the spin-up count. For spin symmetry, it represents the particle count "with id 1".
- qn2::Int: For non-spin symmetry, it represents the spin-down count. For spin symmetry, it represents the particle count "with id 2".
- factor::Float64: A scaling factor, defaults to 1.0
- in_idx::Int: Input index
- op_idx::Int: Operator index
"""
struct VirtSpace
    qn1::Int # For non-spin symmetry, it represents the spin-up count. For spin symmetry, it represents the particle count "with id 1".
    qn2::Int # For non-spin symmetry, it represents the spin-down count. For spin symmetry, it represents the particle count "with id 2".
    factor::Float64
    in_idx::Int
    op_idx::Int
    
    # Constructors
    VirtSpace() = new(0, 0, 1.0, 0, 0)
    VirtSpace(qn1::Int, qn2::Int, factor::Number, in_idx::Int, op_idx::Int) = new(qn1, qn2, Float64(factor), in_idx, op_idx)
end

"""
    apply_op(vs::VirtSpace, op::String, factor::Number, in_idx::Int, op_idx::Int)

Notation: the input virtual space is on the left, so it is *outgoing*.
Therefore, this outputs the input VirtSpace (from the right), such that after applying
the local operator to the physical space, we get the VirtSpace from the left.
This function also set new values for factor, in_idx, and op_idx.
The string can be:
- "I": Identity operator (leaves the VirtSpace unchanged)
- A string of paired characters where each pair consists of:
  - First character: "a" (annihilation) or "c" (creation)
  - Second character: "↑" (spin up) or "↓" (spin down)
  
Examples:
- "a↑": Annihilates a spin-up particle (reduces spinUp by 1).
        The right virtual space will have one less spin-up particle than the left virtual space,
        so that the input (from up) physical space will have one more spin-up particle than
        the output (from down) physical space to mantain the symmetric structure while annihilating a spin-up particle.
- "a↑c↓c↑": Complex sequence of operations, applied from left to right. In total, it increases the VS spinUp count by 1.

Returns a new VirtSpace with the updated values.
"""
function apply_op(vs::VirtSpace, op_str::String, factor::Number, in_idx::Int, op_idx::Int)
    # TODO: For the spin symmetric case, add information to the virtual index about the possible multiplicities. E.g. when op_str==aa (regardless if it's [1,2] or [2,1]), then the output QN (-1,-1) is not sufficient, as it also represents the (0, -2, 1) case, which cannot happen.
    # If identity operator, return a new VirtSpace with updated factor, in_idx, and op_idx
    if op_str == "I"
        return VirtSpace(vs.qn1, vs.qn2, factor, in_idx, op_idx)
    end
    
    # Make a copy to avoid modifying the original
    qn1 = vs.qn1
    qn2 = vs.qn2

    chars = collect(op_str)
    
    # Process the string in pairs
    for i in 1:2:length(chars)
        op = chars[i]
        op_qn = chars[i+1]
        
        if op == 'a'  # Annihilation
            if op_qn == '↑' || op_qn == '1'
                qn1 -= 1
            elseif op_qn == '↓' || op_qn == '2'
                qn2 -= 1
            else
                throw(ArgumentError("Invalid operator character: $op_qn. Expected '↑', '↓', '1', or '2'."))
            end
        elseif op == 'c'  # Creation
            if op_qn == '↑' || op_qn == '1'
                qn1 += 1
            elseif op_qn == '↓' || op_qn == '2'
                qn2 += 1
            else
                throw(ArgumentError("Invalid operator character: $op_qn. Expected '↑', '↓', '1', or '2'."))
            end
        else
            throw(ArgumentError("Invalid operator character: $op. Expected 'a' (annihilation) or 'c' (creation)."))
        end
    end
    
    # Return new VirtSpace with the updated values
    return VirtSpace(qn1, qn2, factor, in_idx, op_idx)
end

# String representation for printing
function Base.show(io::IO, vs::VirtSpace)
    if op_qn1 ∈ ('↑', '↓')
        print(io, "VirtSpace(spinUp=$(vs.qn1), spinDown=$(vs.qn2), factor=$(vs.factor), in_idx=$(vs.in_idx), op_idx=$(vs.op_idx))")
    else
        # For non-spin symmetry, we just print the counts
        print(io, "VirtSpace(qn1=$(vs.qn1), qn2=$(vs.qn2), factor=$(vs.factor), in_idx=$(vs.in_idx), op_idx=$(vs.op_idx))")
    end

end

struct VirtSymmSpace
    qn1::Tuple{Int, Int, Rational{Int64}} # fZ, total e count, total spin
    qn2::Tuple{Int, Int, Rational{Int64}} # fZ, total e count, total spin
    factor::Float64
    in_idx::Int
    op_idx::Int
    
    # Constructors
    VirtSymmSpace() = new((0, 0, 0//1), (0, 0, 0//1), 1.0, 0, 0)
    function VirtSymmSpace(qn1::Tuple{Any, Any, Any}, qn2::Tuple{Any, Any, Any}, factor::Number, in_idx::Int, op_idx::Int)
        (qn1[1] ∈ [0, 1] && qn2[1] ∈ [0, 1]) || throw(ArgumentError("First element of qn must be 0 or 1, as it represents fermionic parity. Got $(qn1[1]) and $(qn2[1])"))
        new((Int(qn1[1]), Int(qn1[2]), convert(Rational{Int64}, qn1[3])), (Int(qn2[1]), Int(qn2[2]), convert(Rational{Int64}, qn2[3])), Float64(factor), in_idx, op_idx)
    end
end

function apply_op(vs::VirtSymmSpace, op_str::String, factor::Number, in_idx::Int, op_idx::Int)
    # If identity operator, return a new VirtSpace with updated factor, in_idx, and op_idx
    if op_str == "I"
        return VirtSymmSpace(vs.qn1, vs.qn2, factor, in_idx, op_idx)
    end
    
    # Make 2-long arrays to access and edit the quantum numbers
    fZ, eCnt, totSpin = collect.(zip(vs.qn1, vs.qn2))

    chars = collect(op_str)
    
    # Process the string in pairs
    for i in 1:2:length(chars)
        op = chars[i]
        op_idx = Int(chars[i+1])

        fZ[op_idx] = fZ[op_idx] == 0 ? 1 : 0  # Fermionic parity is agnostic of the -non I- opetator
        
        if op == 'c'
            eCnt[op_idx] += 1
            totSpin[op_idx] = (totSpin[op_idx] + 1//2) % 1
        elseif op == 'a'
            eCnt[op_idx] -= 1
            totSpin[op_idx] = (totSpin[op_idx] - 1//2) % 1
        end
    end

    qn1, qn2 = zip(fZ, eCnt, totSpin)
    
    # Return new VirtSpace with the updated values
    return VirtSymmSpace(qn1, qn2, factor, in_idx, op_idx)
end

# String representation for printing
function Base.show(io::IO, vs::VirtSymmSpace)
    if isapprox(vs.factor, 1.0) && vs.in_idx == 0 && vs.op_idx == 0
        print(io, "VirtSymmSpace($(vs.qn))")
    else
        print(io, "VirtSymmSpace(quantum number=(fZ=$(vs.qn[1]), total_e_cnt=$(vs.qn[2]), total_spin=$(vs.qn[3])), factor=$(vs.factor), in_idx=$(vs.in_idx), op_idx=$(vs.op_idx))")
    end
end



# Convert OpSum terms to an operator table (similar to Renormalizer's _terms_to_table)
function _terms_to_table(n_sites::Int, terms::OpSum)
    # Initialize a table to store operators
    table = Vector{Vector{Int}}()
    
    # Factor list to store coefficients
    factor_list = Vector{Float64}()
    
    # Track primary operators at each site
    primary_ops_eachsite = [Dict{SiteOp, Int}() for _ in 1:n_sites]
    primary_ops = Vector{SiteOp}()
    
    # Initialize index counter
    index = 1
    
    # Create dummy table entry (identity operator indices for each site)
    # TODO: CHECK: maybe SiteOp does not need to have a site index, as all operators are equally defined at all sites.
    dummy_table_entry = Vector{Int}(undef, n_sites)
    for site in 1:n_sites
        op = SiteOp("I", site)  # Identity operator at this site
        primary_ops_eachsite[site][op] = index
        push!(primary_ops, op)
        dummy_table_entry[site] = index
        index += 1
    end
    
    # Process each term in the OpSum
    for term in terms
        coef = term.coefficient
        site_ops = group_operators_by_site(term)
        
        # Create a copy of the dummy table entry for this term
        table_entry = copy(dummy_table_entry)
        
        # Process each operator in the term
        for (op_str, site) in site_ops
            
            # Create a SiteOp for this operator
            site_op = SiteOp(op_str, site)
            
            # If this operator is not in the primary_ops list for this site, add it
            if !haskey(primary_ops_eachsite[site], site_op)
                primary_ops_eachsite[site][site_op] = index
                push!(primary_ops, site_op)
                index += 1
            end
            
            # Update the table entry for this site
            table_entry[site] = primary_ops_eachsite[site][site_op]
        end
        
        # Add the table entry and factor to our lists
        push!(table, table_entry)
        push!(factor_list, coef)
    end
    
    # Deduplicate table entries and combine factors
    table, factors = _deduplicate_table(table, factor_list)
    
    # Convert table to matrix using stack or vcat/hcat
    table = reduce(vcat, [row' for row in table])  # Using row transpose

    return table, primary_ops, factors
end

function group_operators_by_site(term::OpTerm)
    # Create a dictionary to collect operators at each site
    site_ops_dict = Dict{Int, Vector{String}}()

    if term.operators == ["I"]
        # This represents the nuclear energy term, which acts as an identity operator on every site
        return []
    end
    
    # Group operators by site
    for (op, site) in zip(term.operators, term.sites)
        if !haskey(site_ops_dict, site)
            site_ops_dict[site] = String[]
        end
        push!(site_ops_dict[site], op)
    end
    
    # Create the result array with combined operators for each site
    site_ops = [(join(ops, ""), site) for (site, ops) in site_ops_dict]
    
    return site_ops
end

# Helper function to deduplicate table entries and combine their factors
function _deduplicate_table(table::Vector{Vector{Int}}, factor::Vector{Float64})
    # Create a dictionary to store unique table entries
    unique_entries = Dict{Vector{Int}, Float64}()
    
    # Combine entries with the same operator configuration
    for i in 1:length(table)
        entry = table[i]
        coef = factor[i]
        
        # If the entry already exists, add the coefficient
        if haskey(unique_entries, entry)
            unique_entries[entry] += coef
        else
            unique_entries[entry] = coef
        end
    end
    
    # Convert back to separate arrays
    new_table = collect(keys(unique_entries))
    new_factor = collect(values(unique_entries))
    
    return new_table, new_factor
end


"""
    parity_sign(indexes::Vector{Int}) -> Int

Calculate the parity sign of the permutation required to sort the list of indexes in ascending or descending order.
Each two-site swap is between neighboring sites and contributes a minus sign.

# Arguments
- `indexes::Vector{Int}`: A list of indexes to be sorted.

# Returns
- `Int`: +1 if the number of swaps is even, -1 if the number of swaps is odd.

# Example
```julia
indexes = [3, 1, 4, 2]
sign = parity_sign(indexes)
println("Parity sign: ", sign)  # Output should be -1
"""
function parity_sign(indexes::Vector{Int}; ascending=false)::Int
    # This is a more basic implementation of the original fermionic logic
    indexes_copy = copy(indexes)  # Create a copy to avoid modifying the input
    n = length(indexes_copy)
    sign = +1
    sorted = false
    while !sorted
        sorted = true
        for i in 1:n-1
            flip_cond = ascending ? (indexes_copy[i] > indexes_copy[i+1]) : (indexes_copy[i] < indexes_copy[i+1])
            if flip_cond
                # Swap neighboring elements
                indexes_copy[i], indexes_copy[i+1] = indexes_copy[i+1], indexes_copy[i]
                # Each swap contributes a minus sign
                sign = -sign
                sorted = false
            end
        end
    end

    return sign
end

function parity_sign_symm(indexes::Vector{Int}; ascending=false)::Int
    # This function orders the operator indexes so that they become "aacc".
    # Returns -1 if the count of neighboring swaps are odd, +1 if even.
    @assert length(indexes) == 4 "parity_sign_symm expects a vector of length 4, got $(length(indexes))"
    current_op_order = ["a", "a", "c", "c"]
    
    indexes_copy = copy(indexes)  # Create a copy to avoid modifying the input
    n = length(indexes_copy)
    sign = +1
    sorted = false
    while !sorted
        sorted = true
        for i in 1:n-1
            if current_op_order[i] == current_op_order[i+1]
                continue # We do not distinguish between two creation/annihilation operators for the operator's coefficient sign
            end
            order_cond = ascending ? (indexes_copy[i] < indexes_copy[i+1]) : (indexes_copy[i] > indexes_copy[i+1])
            if order_cond
                # Swap neighboring elements
                indexes_copy[i], indexes_copy[i+1] = indexes_copy[i+1], indexes_copy[i]
                # Update the operator order
                current_op_order[i], current_op_order[i+1] = current_op_order[i+1], current_op_order[i]
                # Each swap contributes a minus sign
                sign = -sign
                sorted = false
            end
        end
    end

    return sign
end

gen_OpSum(chem_data::ChemProperties, ord::Vector{Int}; kwargs...) = gen_OpSum(chem_data::ChemProperties; ord=ord, kwargs...)
gen_OpSum(chem_data::ChemProperties; n_sites=0, ord=nothing, tol=1e-14, spin_symm::Bool=false, add_nuc::Bool=true) = gen_OpSum(chem_data.h1e, chem_data.h2e, chem_data.e_nuc; n_sites=n_sites, ord=ord, tol=tol, spin_symm=spin_symm, add_nuc=add_nuc)
gen_OpSum(h1e, h2e, e_nuc; kwargs...) = gen_OpSum(h1e, h2e; e_nuc=e_nuc, kwargs...)
gen_OpSum(h1e, h2e, e_nuc, ord::Vector{Int}; kwargs...) = gen_OpSum(h1e, h2e; e_nuc=e_nuc, ord=ord, kwargs...)
gen_OpSum(h1e, h2e, e_nuc, n_sites::Int; kwargs...) = gen_OpSum(h1e, h2e; e_nuc=e_nuc, n_sites=n_sites, kwargs...)

function gen_OpSum(h1e::AbstractArray{Float64}, h2e::AbstractArray{Float64}; e_nuc=0.0, n_sites=0, ord=nothing, tol=1e-14, spin_symm::Bool=false, add_nuc::Bool=true)
    add_nuc || (e_nuc = 0.0)  # If add_nuc is false, set nuclear energy to 0
    if isnothing(ord)
        if n_sites > 0
            @assert size(h1e, 1) >= n_sites "Provided h1e size $(size(h1e)) does not match n_sites $(n_sites)"
            ord = collect(1:n_sites)  # Default ordering if n_sites is specified
        else
            ord = collect(1:size(h1e, 1))  # Default ordering if no specific ordering is provided
        end
    else
        # Ensure ord is a valid vector of integers
        @assert length(ord) >= n_sites "Provided ord length $(length(ord)) is lower than n_sites $(n_sites)"
        (n_sites > 0) && (ord = ord[1:n_sites])  # Truncate or use the provided order
    end

    # Generate the OpSum object from the Hamiltonian coefficients
    if spin_symm
        # return _gen_OpSum_SpinSymm(h1e, h2e, e_nuc, ord; tol=tol) # Unprocessed h1e and h2e
        return _gen_OpSum_SpinSymm_processed(h1e, h2e, e_nuc, ord; tol=tol) # Always assume that h1e and h2e are already processed when they are passed to gen_OpSum?
    else
        # return _gen_OpSum_noSpinSymm(h1e, h2e, e_nuc, ord; tol=tol)
        return _gen_OpSum_noSpinSymm_processed(h1e, h2e, e_nuc, ord; tol=tol)
    end
end

# Generate the OpSum object from the Hamiltonian coefficients:
function _gen_OpSum_SpinSymm_processed(h1e, h2e, e_nuc, ord; tol=1e-14)
    N_spt = length(ord)

    os = OpSum()

    # Nuclear energy term
    if e_nuc != 0.0
        # Add the nuclear energy term as an identity operator on all sites
        os += e_nuc, ["I"], [0]
    end

    # One-electron terms
    for p = 1:N_spt, q = 1:N_spt
        cf = h1e[ord[p], ord[q]] #* parity_sign([p, q])

        if abs(cf) >= tol
            if p + q - 1 <= N_spt # TODO: Check if it improves performance because of the bipartite grouping 
                os += cf, "a1", p, "c1", q  # Spin-up operators
            else
                os += cf, "a2", p, "c2", q  # Spin-down operators
            end
        end
    end

    # Two-electron terms
    for p = 1:N_spt, q = 1:N_spt, r = 1:N_spt, s = 1:N_spt
        # According to the h2e defition, h2e[i, j, k, l] represents ⟨ij‖kl⟩, so i becomes c_i (creation in orbital i), j is c_j, k is a_k and l is a_l.

        # cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign_symm([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
        # cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
        cf = h2e[ord[p], ord[q], ord[r], ord[s]] # * parity_sign([p, q, r, s]) # is it the same as parity_sign([p, q, r, s])?

        if abs(cf) >= tol
            os += cf, "a1", p, "a2", q, "c2", r, "c1", s

            # if (p + q) <= (r + s) # Keep the lower indexes on the left side in order to match with the single hopping terms
            #     # TODO: Why p, r, s, q order?
            #     os += cf, "a1", p, "a2", r, "c2", s, "c1", q
            # else
            #     os += cf, "a2", p, "a1", r, "c1", s, "c2", q
            # end
        end
    end

    # # HALF Two-electron terms
    # # for p = 1:N_spt, q = 1:N_spt, r = 1:N_spt, s = 1:N_spt
    # for p = 1:N_spt, r = p:N_spt, s = 1:N_spt, q = 1:N_spt
    #     # TODO: Why p, r, s, q order? This swaps the original operators coefficients symmetrically by interchanging one creation with one annihilation operator.
    #     # According to the h2e defition, h2e[i, j, k, l] represents ⟨ij‖kl⟩, so i becomes c_i (creation in orbital i), j is c_j, k is a_k and l is a_l.

    #     # cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign_symm([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
    #     cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
    #     # cf = h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?

    #     if abs(cf) >= tol
    #         os += cf, "a1", p, "a2", r, "c2", s, "c1", q

    #         # if (p + q) <= (r + s) # Keep the lower indexes on the left side in order to match with the single hopping terms
    #         #     # TODO: Why p, r, s, q order?
    #         #     os += cf, "a1", p, "a2", r, "c2", s, "c1", q
    #         # else
    #         #     os += cf, "a2", p, "a1", r, "c1", s, "c2", q
    #         # end
    #     end
    # end

    return os
end


# Generate the OpSum object from the Hamiltonian coefficients:
function _gen_OpSum_SpinSymm(h1e, h2e, e_nuc, ord; tol=1e-14)
    N_spt = length(ord)

    os = OpSum()

    # Nuclear energy term
    if e_nuc != 0.0
        # Add the nuclear energy term as an identity operator on all sites
        os += e_nuc, ["I"], [0]
    end

    # One-electron terms
    for p = 1:N_spt, q = 1:N_spt
        cf = h1e[ord[p], ord[q]] * parity_sign([p, q])

        if abs(cf) >= tol
            if p + q - 1 <= N_spt # TODO: Check if it improves performance because of the bipartite grouping 
                os += cf, "a1", p, "c1", q  # Spin-up operators
            else
                os += cf, "a2", p, "c2", q  # Spin-down operators
            end
        end
    end

    # Two-electron terms
    for p = 1:N_spt, q = 1:N_spt, r = 1:N_spt, s = 1:N_spt
        # TODO: Why p, r, s, q order?
        # According to the h2e defition, h2e[i, j, k, l] represents ⟨ij‖kl⟩, so i becomes c_i (creation in orbital i), j is c_j, k is a_k and l is a_l.

        # cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign_symm([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
        cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
        # cf = h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?

        if abs(cf) >= tol
            # os += cf, "a1", p, "a2", r, "c2", s, "c1", q

            if (p + q) <= (r + s) # Keep the lower indexes on the left side in order to match with the single hopping terms
                # TODO: Why p, r, s, q order?
                os += cf, "a1", p, "a2", r, "c2", s, "c1", q
            else
                os += cf, "a2", p, "a1", r, "c1", s, "c2", q
            end
        end
    end

    # # HALF Two-electron terms
    # # for p = 1:N_spt, q = 1:N_spt, r = 1:N_spt, s = 1:N_spt
    # for p = 1:N_spt, r = p:N_spt, s = 1:N_spt, q = 1:N_spt
    #     # TODO: Why p, r, s, q order? This swaps the original operators coefficients symmetrically by interchanging one creation with one annihilation operator.
    #     # According to the h2e defition, h2e[i, j, k, l] represents ⟨ij‖kl⟩, so i becomes c_i (creation in orbital i), j is c_j, k is a_k and l is a_l.

    #     # cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign_symm([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
    #     cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
    #     # cf = h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?

    #     if abs(cf) >= tol
    #         os += cf, "a1", p, "a2", r, "c2", s, "c1", q

    #         # if (p + q) <= (r + s) # Keep the lower indexes on the left side in order to match with the single hopping terms
    #         #     # TODO: Why p, r, s, q order?
    #         #     os += cf, "a1", p, "a2", r, "c2", s, "c1", q
    #         # else
    #         #     os += cf, "a2", p, "a1", r, "c1", s, "c2", q
    #         # end
    #     end
    # end

    return os
end

function _gen_OpSum_noSpinSymm_processed(h1e, h2e, e_nuc, ord; tol=1e-14)
    N_spt = length(ord)

    os = OpSum()

    # Nuclear energy term
    if e_nuc != 0.0
        # Add the nuclear energy term as an identity operator on all sites
        os += e_nuc, ["I"], [0]
    end

    # One-electron terms
    for p = 1:N_spt, q = 1:N_spt
        cf = h1e[ord[p], ord[q]] #* parity_sign([p, q])

        if abs(cf) >= tol
            os += cf, "a↑", p, "c↑", q
            os += cf, "a↓", p, "c↓", q
        end
    end

    # Two-electron terms
    for p = 1:N_spt, q = 1:N_spt, r = 1:N_spt, s = 1:N_spt
        # cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?
        cf = h2e[ord[p], ord[q], ord[r], ord[s]] # * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?

        if abs(cf) >= tol
            os += cf, "a↑", p, "a↓", q, "c↓", r, "c↑", s
            os += cf, "a↓", p, "a↑", q, "c↑", r, "c↓", s
            if p != r && s != q
                os += cf, "a↓", p, "a↓", q, "c↓", r, "c↓", s
                os += cf, "a↑", p, "a↑", q, "c↑", r, "c↑", s
            end
        end
    end

    return os
end


function _gen_OpSum_noSpinSymm(h1e, h2e, e_nuc, ord; tol=1e-14)
    N_spt = length(ord)

    os = OpSum()

    # Nuclear energy term
    if e_nuc != 0.0
        # Add the nuclear energy term as an identity operator on all sites
        os += e_nuc, ["I"], [0]
    end

    # One-electron terms
    for p = 1:N_spt, q = 1:N_spt
        cf = h1e[ord[p], ord[q]] * parity_sign([p, q])

        if abs(cf) >= tol
            os += cf, "a↑", p, "c↑", q
            os += cf, "a↓", p, "c↓", q
        end
    end

    # Two-electron terms
    for p = 1:N_spt, q = 1:N_spt, r = 1:N_spt, s = 1:N_spt
        cf = 0.5 * h2e[ord[p], ord[q], ord[r], ord[s]] * parity_sign([p, r, s, q]) # is it the same as parity_sign([p, q, r, s])?

        if abs(cf) >= tol
            os += cf, "a↑", p, "a↓", r, "c↓", s, "c↑", q
            os += cf, "a↓", p, "a↑", r, "c↑", s, "c↓", q
            if p != r && s != q
                os += cf, "a↓", p, "a↓", r, "c↓", s, "c↓", q
                os += cf, "a↑", p, "a↑", r, "c↑", s, "c↑", q
            end
        end
    end

    return os
end

"""
    construct_symbolic_mpo(n_sites::Int, table, primary_ops, factor; algo="Hungarian")

Construct a symbolic MPO representation optimized for bond dimension.
This follows the approach from Renormalizer.

Parameters:
- n_sites: Number of sites
- table: Table of operator indices for each term
- primary_ops: List of primary operators (SiteOp objects)
- factor: Coefficients for each term in the table
- algo: Algorithm to use for optimization ("Hungarian" is implemented, "Hopcroft-Karp" will be implemented in the future)

Returns:
- symbolic_mpo: A representation of the MPO with optimized structure


The idea:

the index of the primary ops {0:"I", 1:"a", 2:r"a^†"}

for example: H = 2.0 * a_1 a_2^dagger   + 3.0 * a_2^† a_3 + 4.0*a_0^† a_3
The column names are the site indices with 0 and 4 imaginary (see the note below)
and the content of the table is the index of primary operators.
                    s0   s1   s2   s3  s4  factor
a_1 a_2^dagger      0    1    2    0   0   2.0
a_2^† a_3     0    0    2    1   0   3.0
a_1^† a_3     0    2    0    1   0   4.0
for convenience the first and last column mean that the operator of the left and right hand of the system is I

cut the string to construct the row(left) and column(right) operator and find the duplicated/independent terms
                    s0   s1 |  s2   s3  s4 factor
a_1 a_2^dagger      0    1  |  2    0   0  2.0
a_2^† a_3     0    0  |  2    1   0  3.0
a_1^† a_3     0    2  |  0    1   0  4.0

 The content of the table below means matrix elements with basis explained in the notes.
 In the matrix elements, 1 means combination and 0 means no combination.
      (2,0,0) (2,1,0) (0,1,0)  -> right side of the above table
 (0,1)   1       0       0
 (0,0)   0       1       0
 (0,2)   0       0       1
   |
   v
 left side of the above table
 In this case all operators are independent so the content of the matrix is diagonal

and select the terms and rearrange the table
The selection rule is to find the minimal number of rows+cols that can eliminate the
matrix
                  s1   s2 |  s3 s4 factor
a_1 a_2^dagger    0'   2  |  0  0  2.0
a_2^† a_3   1'   2  |  1  0  3.0
a_1^† a_3   2'   0  |  1  0  4.0
0'/1'/2' are three new operators(could be non-elementary)
The local mpo is the transformation matrix between 0,0,0 to 0',1',2'.
In this case, the local mpo is simply (1, 0, 2)

cut the string and find the duplicated/independent terms
        (0,0), (1,0)
 (0',2)   1      0
 (1',2)   0      1
 (2',0)   0      1

and select the terms and rearrange the table
apparently choose the (1,0) column and construct the complementary operator (1',2)+(2',0) is better
0'' =  3.0 * (1', 2) + 4.0 * (2', 0)
                                             s2     s3 | s4 factor
(4.0 * a_1^dagger + 3.0 * a_2^dagger) a_3    0''    1  | 0  1.0
a_1 a_2^dagger                               1''    0  | 0  2.0
0''/1'' are another two new operators(non-elementary)
The local mpo is the transformation matrix between 0',1',2' to 0'',1''

         (0)
 (0'',1)  1
 (1'',0)  1

The local mpo is the transformation matrix between 0'',1'' to 0'''

"""
function construct_symbolic_mpo(table, primary_ops, factor; algo="Hungarian", verbose=true)

    n_sites = size(table, 2)
    
    # Add ones at the beginning and end of each row. They will be the index of the trivial auxiliary virtual bonds at the start and end of the MPO
    ones_col = ones(Int, size(table, 1))
    table = hcat(ones_col, table, ones_col)
    
    # Start with the trivial virtual space.
    # In Renormalizer is it just a list with length 1 and the list is actually in the first index
    virtSpace_in = [[VirtSpace()]]
    
    # Store the list of virtual spaces at each site
    virtSpace_out_list = [virtSpace_in]

    verbose && println("Using $(algo) algorithm for bipartite matching optimization")
    
    # This is the main loop in Renormalizer's construct_symbolic_mpo
    for isite in 1:n_sites
        verbose && println("Processing site $(isite) of $(n_sites)")
        # Split table into row and column parts - always take first two columns
        # for rows and the rest for columns, as in Renormalizer
        table_row = table[:, 1:2]
        table_col = table[:, 3:end]
        
        # Call the one_site function to process this site
        virtSpace_out, table, factor = _construct_symbolic_mpo_one_site(
            table_row, table_col, virtSpace_in, factor, primary_ops; algo=algo
        )
        
        # Update for next iteration
        virtSpace_in = virtSpace_out
        
        # Store the virtual spaces for this site
        push!(virtSpace_out_list, virtSpace_out)
    end
    
    # At the end, we should have a single term with factor 1 (or close to it due to floating point)
    # Comment out assert for now during development
    @assert size(table, 1) == length(factor) == 1 && isapprox(factor[1], 1.0, atol=1e-10)
    @assert length(virtSpace_out_list) == n_sites + 1

    # Construct symbolic MPO
    symbolic_mpo = [] # Preallocate if possible
    for isite in 1:n_sites
        symbolic_site = compose_symbolic_site_sparse(virtSpace_out_list[isite], virtSpace_out_list[isite+1], primary_ops)
        push!(symbolic_mpo, symbolic_site)
    end

    # Calculate the virtual spaces' quantum numbers

    mpoVs = Vector{Vector{Vector{Tuple{Int, Int}}}}(undef, length(virtSpace_out_list))
    for (i, virtSpace_out) in enumerate(virtSpace_out_list)
        mpoVs[i] = [[(vs.qn1, vs.qn2) for vs in vs_grp] for vs_grp in virtSpace_out]
    end

    @assert all(length(unique(vs))==1 for vs_grp in mpoVs for vs in vs_grp)
    verbose && println("symbolic MPO's bond dimensions: $([length(vs) for vs in mpoVs])")
    
    return symbolic_mpo, mpoVs
end

"""
    _construct_symbolic_mpo_one_site(table_row, table_col, virtSpace_in_list, factor, primary_ops; algo="Hungarian")

Process one site in the MPO construction, following Renormalizer's approach.

Parameters:
- table_row: Left part of the table
- table_col: Right part of the table
- virtSpace_in_list: Input virtual spaces from previous site
- factor: Coefficients for each term
- primary_ops: List of primary operators
- algo: Algorithm to use ("Hungarian" is implemented, "Hopcroft-Karp" will be implemented in the future)
- k: Extra parameter (usually 0)

Returns:
- virtSpaces_out: Output virtual spaces for this site
- new_table: Updated table for next site
- new_factor: Updated factors
"""
function _construct_symbolic_mpo_one_site(table_row, table_col, virtSpace_in, factor, primary_ops; algo="Hungarian")
    # Find unique rows and their inverse mapping
    term_row, row_unique_inverseMap = find_unique_with_inverseMap(table_row)
    
    # Make sure the dimensions match
    @assert size(table_row, 2) == 2
    
    # Find unique columns and their inverse mapping
    term_col, col_unique_inverseMap = find_unique_with_inverseMap(table_col)
    
    # Create a sparse matrix directly where non-zero values are indices into the factor array
    non_red = sparse(row_unique_inverseMap, col_unique_inverseMap, 1:length(factor))
    
    return _decompose_graph(term_row, term_col, non_red, virtSpace_in, factor, primary_ops, algo)
end

"""
    _decompose_graph(term_row, term_col, non_red, virtSpace_in_list, factor, primary_ops, algo)

Implement the graph-based optimization of MPO representation using bipartite matching.
This is based on Renormalizer's implementation which uses a bipartite vertex cover
approach to find the minimal set of rows and columns that cover all non-zero elements.

Parameters:
- term_row: Unique rows from the table
- term_col: Unique columns from the table
- non_red: Non-redundant sparse matrix
- virtSpace_in_list: Input virtual spaces from previous site
- factor: Coefficients for each term
- primary_ops: List of primary operators
- algo: Algorithm to use ("Hungarian" is implemented, "Hopcroft-Karp" will be implemented in the future)
- k: Extra parameter (usually 0)

Returns:
- virtSpaces_out: Output virtual spaces for this site
- new_table: Updated table for next site
- new_factor: Updated factors

Note: The Hungarian algorithm tends to produce more optimal MPO bond dimensions
      for quantum chemistry Hamiltonians, while "Hopcroft-Karp" is more time efficient.
"""
function _decompose_graph(term_row, term_col, non_red, virtSpace_in, factor, primary_ops, algo)
    
    # Get dimensions directly from the sparse matrix
    n_rows, n_cols = size(non_red)
    
    # Use transpose to convert to CSC format and exploit efficient row access
    # Previously, `if !isempty(row_select)` was used to check if non_red_T was needed. it might overallocate memory.
    non_red_T = sparse(transpose(non_red))  # Transpose converts CSC to CSR (effectively). TODO: Define it directly in CSR format (:rows as last argument) here by merging the two functions and having access to the inverseMaps. This allocates new data, so it might be better to calculate this in the caller function.
    sparse_rows_T = rowvals(non_red_T)
    sparse_vals_T = nonzeros(non_red_T)

    # More efficient way to build the bigraph based on matrix dimensions
    if n_rows < n_cols
        
        # Note rows (columns) are columns (rows) in the original matrix
        bigraph = [sparse_rows_T[nzrange(non_red_T, row)] for row in 1:n_rows]
        
        # Compute vertex cover using the specified algorithm
        rowbool, colbool = compute_bipartite_vertex_cover(bigraph, algo)
    else

        sparse_rows = rowvals(non_red)
        bigraph = [sparse_rows[nzrange(non_red, row)] for row in 1:n_cols]
        
        # When rows >= cols, swap the results to match Renormalizer
        colbool, rowbool = compute_bipartite_vertex_cover(bigraph, algo)
    end

    # Find selected rows and columns based on the vertex cover
    row_select = findall(rowbool)
    # Sort row_select by how many columns each row covers, largest cover first
    # This helps optimize the MPO bond dimension
    sort!(row_select, by=row -> -length(nzrange(non_red_T, row)))

    col_select = findall(colbool)
    

    # Initialize output virtual spaces, tables, and factors
    virtSpaces_out = [] # Vector{Vector{VirtSpace}}(undef, length(virtSpace_in_list)) ?
    new_table = []
    new_factor = []

    # Process selected rows first (following Renormalizer's approach)
    for row_idx in row_select
        # Create an output virtual space for this row
        symbol = term_row[row_idx]
        
        # Calculate quantum numbers using input virtual spaces and primary operators
        # If symbol has quantum number information, we would use it here
        # In Renormalizer's version, this would compute quantum numbers based on the row symbol
        # For now, we use (0,0) as default quantum numbers
        # virtSpace_out = _compute_virtSpace(virtSpace_in_list, symbol, primary_ops)
        @assert length(unique((v.qn1, v.qn2) for v in virtSpace_in[symbol[1]])) == 1 # Remove when the code is stable
        
        virtSpace_out = apply_op(virtSpace_in[symbol[1]][1], primary_ops[symbol[2]].operator, 1.0, symbol[1], symbol[2])
        
        # Add the new virtual space to our list
        push!(virtSpaces_out, [virtSpace_out])

        rows_range = nzrange(non_red_T, row_idx)

        table_entry_n_rows = length(rows_range)
        table_entry_n_cols = length(term_col[1]) + 1

        virtSpace_next_idx = length(virtSpaces_out)

        table_entry = Matrix{Int}(undef, table_entry_n_rows, table_entry_n_cols)
        for (i, sparse_row_idx) in enumerate(sparse_rows_T[rows_range])
            table_entry[i, 1] = virtSpace_next_idx
            table_entry[i, 2:table_entry_n_cols] = term_col[sparse_row_idx]
        end
        push!(new_table, table_entry)

        append!(new_factor, factor[sparse_vals_T[rows_range]])
        

        sparse_vals_T[rows_range] .= 0
                
    end

    
    dropzeros!(non_red_T)
    non_red = sparse(non_red_T')
    
    # Process selected columns
    for col_idx in col_select
        # Create a multi-operator entry for this column
        push!(virtSpaces_out, [])

        for (row_idx, val) in zip(findnz(non_red[:, col_idx])...)
            symbol = term_row[row_idx]
            @assert length(unique((v.qn1, v.qn2) for v in virtSpace_in[symbol[1]])) == 1 # Remove when the code is stable
            virtSpace_out = apply_op(virtSpace_in[symbol[1]][1], primary_ops[symbol[2]].operator, factor[val], symbol[1], symbol[2]) # In Renormalizer: factor=factor[non_red_one_col[i] - 1]. Why the -1? Indexing should match the original table. TODO: Also, make sure the factor is correct.
            push!(virtSpaces_out[end], virtSpace_out)
        end
        push!(new_table, reshape(vcat([length(virtSpaces_out)], term_col[col_idx]), 1, :))
        push!(new_factor, 1.0)
    end
    
    table = vcat(new_table...)

    @assert size(table, 1) == length(new_factor) "Table length ($(length(table))) does not match factor length ($(length(new_factor)))"

    return virtSpaces_out, table, new_factor

end

"""
    bipartite_vertex_cover_from_matchingV(bigraph, matchingV)

Compute a minimum vertex cover of a bipartite graph from a maximum matching.
This is based on König's theorem, which states that in a bipartite graph, 
the size of a minimum vertex cover equals the size of a maximum matching.

Parameters:
- bigraph: Bipartite graph as an array of arrays where bigraph[u] contains all vertices v that 
           are connected to u (adjacency list format)
- matching: Maximum matching as an array where matchingV[v] = u if u and v are matched

Returns:
- rowbool: Array of booleans indicating which rows are in the vertex cover
- colbool: Array of booleans indicating which columns are in the vertex cover
"""
function bipartite_vertex_cover_from_matchingV(bigraph, matchingV)
    # Initialize visit arrays
    nU = length(bigraph)
    nV = length(matchingV)
    
    # Generate the matching from right to left
    matchingU = Vector{Union{Int, Nothing}}(nothing, nU)

    for (v, u) in enumerate(matchingV)
        if u !== nothing
            matchingU[u] = v
        end
    end
    

    # Implementation of König's theorem Algorithm:
    # 1. Start with unmatched vertices in U (left set)
    # 2. Build an alternating tree using DFS
    # 3. The minimum vertex cover consists of:
    #    - Vertices in U that are NOT visited
    #    - Vertices in V that are visited
    
    # Use iterative implementation to avoid recursion depth issues with large graphs
    # This is equivalent to Renormalizer's new_konig() function (wait_u = set(range(nU)) - set(matchV))
    visitU = fill(false, nU)
    visitV = fill(false, nV)
    
    # Create a set of all vertices in U first
    all_vertices = Set(1:nU)
    
    # Create a set of all matched vertices in U (non-nothing values)
    matched_vertices = Set(u for u in matchingV if u !== nothing)
    
    # Find unmatched vertices by set difference (more efficient)
    wait_u = setdiff(all_vertices, matched_vertices)
    
    # Build alternating tree using BFS (breadth-first search)
    while !isempty(wait_u)
        u = pop!(wait_u)
        visitU[u] = true
        
        # Visit all neighbors of u
        for v in bigraph[u]
            if !visitV[v]
                visitV[v] = true
                # The vertex must be matched (otherwise matching wouldn't be maximum)
                match_u = matchingV[v]
                @assert match_u !== nothing
                @assert !visitU[match_u]
                push!(wait_u, match_u)
            end
        end
    end
    
    # Generate the vertex cover
    # König's theorem: vertices not in visitU and vertices in visitV form a minimum vertex cover
    rowbool = [!visit for visit in visitU]
    colbool = visitV
    
    return rowbool, colbool
end

"""
    alternate_tree_dfs(u, bigraph, visitU, visitV, matching)

Perform depth-first search to build an alternating tree from an unmatched vertex.
This is a key part of the König's theorem algorithm to find a minimum vertex cover.

Parameters:
- u: Current vertex in the left set (U)
- bigraph: Bipartite graph as an array of arrays where bigraph[u] contains all vertices v
- visitU: Boolean array marking visited vertices in U
- visitV: Boolean array marking visited vertices in V
- matching: Maximum matching where matching[v] = u if u and v are matched

No return value, modifies visitU and visitV in place.
"""
function alternate_tree_dfs(u, bigraph, visitU, visitV, matching)
    visitU[u] = true
    
    # Visit all neighbors of u
    for v in bigraph[u]
        if !visitV[v]
            visitV[v] = true
            
            # The vertex must be matched (otherwise matching wouldn't be maximum)
            match_u = matching[v]
            
            if match_u !== nothing
                # Continue DFS from the matched vertex
                alternate_tree_dfs(match_u, bigraph, visitU, visitV, matching)
            end
        end
    end
end


"""
    elementary_operator(op_type::String, phys_space::GradedSpace, dataType::DataType=Float64)

Create an elementary operator tensor for a given operator type.
This is a helper function for convert_to_tensorkit_mpo.

Parameters:
- op_type: String representing the operator ("a↑", "c↓", "I", etc.)
- phys_space: Physical space for the tensor
- dataType: Data type for tensor elements

Returns:
- TensorMap representing the elementary operator
"""
function elementary_operator(op_type::String, phys_space::GradedSpace, dataType::DataType=Float64)
    # Create tensor space for the operator
    tensor_space = phys_space ⊗ phys_space'
    
    # Initialize tensor with zeros
    op_tensor = TensorMap(zeros, dataType, tensor_space)
    
    # Fill tensor based on operator type
    if op_type == "I"
        # Identity operator
        for i in 1:dim(phys_space)
            op_tensor[i, i] = 1.0
        end
    elseif op_type == "a↑"
        # Annihilation operator for spin up
        # Implement based on fermionic operators in TensorKit
        # This is a placeholder - actual implementation depends on the conventions used
    elseif op_type == "c↑"
        # Creation operator for spin up
    elseif op_type == "a↓"
        # Annihilation operator for spin down
    elseif op_type == "c↓"
        # Creation operator for spin down
    elseif op_type == "n↑"
        # Number operator for spin up
    elseif op_type == "n↓"
        # Number operator for spin down
    else
        # Handle composite operators or custom operators
        # Parse the operator string and create the appropriate tensor
    end
    
    return op_tensor
end

"""
    find_unique_with_inverse(arrays)

Find unique rows in a collection of arrays and return the inverse mapping.
This is equivalent to NumPy's `np.unique(arrays, axis=0, return_inverse=True)`.

Parameters:
- arrays: An array of arrays (like table_row or table_col)

Returns:
- unique_arrays: List of unique arrays
- inverse_mapping: Array where each element i contains the index of arrays[i] in unique_arrays
"""
function find_unique_with_inverseMap(arrays)
    # Handle different input types
    if isa(arrays, Matrix)
        # Input is already a matrix
        matrix_arrays = arrays
    elseif isa(arrays, Vector) && all(isa.(arrays, Vector))
        # Input is a vector of vectors, convert to matrix
        matrix_arrays = reduce(vcat, [row' for row in arrays])
    else
        # Handle other cases or throw error
        error("Input must be a matrix or vector of vectors")
    end
    
    unique_arrays = Vector{Vector{Int}}()
    inverse_mapping = Int[]
    
    # Use Dict to track unique arrays
    lookup = Dict{Any, Int}()
    
    for arr in eachrow(arrays)
        # Convert to tuple for hashing in Dict
        key = Tuple(arr)
        
        if !haskey(lookup, key)
            # Found a new unique array
            push!(unique_arrays, copy(arr))
            lookup[key] = length(unique_arrays)
        end
        
        # Record the position in unique_arrays
        push!(inverse_mapping, lookup[key])
    end
    
    return unique_arrays, inverse_mapping
end

"""
    hungarian_max_bipartite_matching(bigraph)

Implement the Hungarian algorithm for bipartite matching, which finds an optimal
assignment between two sets of nodes in a bipartite graph.
This is ported from Renormalizer's Python implementation (max_bipartite_matching2).

Parameters:
- bigraph: A bipartite graph represented as an array of arrays where bigraph[u] contains 
           all vertices v that are connected to u (adjacency list format)

Returns:
- An array where each element at index v contains the matched left node u for right node v,
  or nothing if the right node is unmatched
"""
function hungarian_max_bipartite_matching(bigraph)
    # Find the max index in bigraph values to determine nV
    nU = length(bigraph)
    nV = 0
    for adjlist in bigraph
        if !isempty(adjlist)
            nV = max(nV, maximum(adjlist))
        end
    end
    
    # Initialize match array for vertices in V with size nV
    # In Julia indices start at 1, but we need to access the index with the max value found
    match = Vector{Union{Int, Nothing}}(undef, nV)
    fill!(match, nothing)
    
    function augment(u, visit)
        # Try to find augmenting path starting at u
        for v in bigraph[u]
            if !visit[v]
                visit[v] = true
                if match[v] === nothing || augment(match[v], visit)
                    match[v] = u  # Found an augmenting path
                    return true
                end
            end
        end
        return false
    end
    
    # Try to augment each unmatched vertex in U
    for u in 1:nU
        augment(u, fill(false, nV))
    end
    
    return match
end

"""
    process_matching_result(row_idx, col_idx, factor_idx, term_row, term_col, virtSpace_in_list, factor, primary_ops, virtSpaces_out, new_factors)

Helper function to process a matching result and create the corresponding site operator.
This creates a site operator that combines the input virtual spaces from the previous step
and the primary operators for the current site.

Parameters:
- row_idx: Index of row in term_row
- col_idx: Index of column in term_col
- factor_idx: Index in factor array
- term_row: Unique rows from the table
- term_col: Unique columns from the table
- virtSpace_in_list: Input virtual spaces from previous site
- factor: Coefficients for each term
- primary_ops: List of primary operators
- virtSpaces_out: Output virtual spaces list to be modified
- new_factors: New factors list to be modified

No return value, modifies virtSpaces_out and new_factors in place.
"""
function process_matching_result(row_idx, col_idx, factor_idx, term_row, term_col, virtSpace_in_list, factor, primary_ops, virtSpaces_out, new_factors)
    current_factor = factor[factor_idx]
    
    # Extract the relevant virtual space information
    virtSpace_in_pos = term_row[row_idx][1:length(virtSpace_in_list)]
    virtSpace_out_pos = term_col[col_idx]
    
    # Create the site operator (a list of SiteOp objects)
    site_op = SiteOp[]
    
    # Add input operators (connect to previous site's operators)
    for i in 1:length(virtSpace_in_list)
        if virtSpace_in_pos[i] > 0
            # Look up the input operator and create a new SiteOp with its operator type
            virtSpace_in = virtSpace_in_list[i]
            if typeof(virtSpace_in) <: SiteOp
                # Just copy the operator type and site, but keep coefficient as 1.0
                # because the coefficient is handled separately
                push!(site_op, SiteOp(1.0, virtSpace_in.operator, virtSpace_in.site))
            else
                # If virtSpace_in_list contains operator indices rather than SiteOp objects,
                # we need to look up the operator in primary_ops
                push!(site_op, SiteOp(1.0, primary_ops[virtSpace_in_pos[i]].operator, primary_ops[virtSpace_in_pos[i]].site))
            end
        end
    end
    
    # Add primary operators for this site
    for j in 1:length(virtSpace_out_pos)
        if virtSpace_out_pos[j] > 0
            # Create a new SiteOp for the primary operator
            push!(site_op, SiteOp(1.0, primary_ops[virtSpace_out_pos[j]].operator, primary_ops[virtSpace_out_pos[j]].site))
        end
    end
    
    # Add the site operator and factor to the output lists
    push!(virtSpaces_out, site_op)
    push!(new_factors, current_factor)
end

"""
    compute_bipartite_vertex_cover(bigraph, algo)

Compute a minimum vertex cover of a bipartite graph using the specified algorithm.
This mimics Renormalizer's bipartite_vertex_cover function.

Parameters:
- bigraph: A bipartite graph represented as an adjacency list
- algo: The algorithm to use ("Hungarian" is currently the only supported option)

Returns:
- rowbool: Array of booleans indicating which rows are in the vertex cover
- colbool: Array of booleans indicating which columns are in the vertex cover
"""
function compute_bipartite_vertex_cover(bigraph, algo)
    if algo == "Hungarian"
        # Find maximum matching first using the Hungarian algorithm
        matchingV = hungarian_max_bipartite_matching(bigraph)
        
        # Compute minimum vertex cover from maximum matching using König's theorem
        return bipartite_vertex_cover_from_matchingV(bigraph, matchingV)
    elseif algo == "Hopcroft-Karp"
        error("Hopcroft-Karp algorithm not yet implemented")
    else
        error("Unsupported algorithm: '$(algo)'. Only 'Hungarian' is currently supported.")
    end
end

"""
    compose_symbolic_site(virtSpace_in, virtSpace_out, primary_ops)

Compose a symbolic Matrix Operator for a site in the MPO.

Parameters:
- virtSpace_in: Input virtual spaces
- virtSpace_out: Output virtual spaces
- primary_ops: List of primary operators (Mapping: SiteOp objects to index)

Returns:
- A matrix of operator lists representing the MPO tensor for this site
"""
function compose_symbolic_site(virtSpace_in, virtSpace_out, primary_ops)
    # Create a matrix of empty SiteOp arrays with dimensions [length(virtSpace_in) × length(virtSpace_out)]
    mo = [SiteOp[] for _ in 1:length(virtSpace_in), _ in 1:length(virtSpace_out)]
    
    # For each output operator
    for (ivs, vs_out) in enumerate(virtSpace_out)
        # For each composed operator in this output operator
        for op in vs_out
            # Get the input index and operator from the composed operator's symbol
            site_op = primary_ops[op.op_idx]
            
            # Multiply the operator by the factor and add it to the appropriate cell
            push!(mo[op.in_idx, ivs], site_op * op.factor)
        end
    end
    
    return mo
end


"""
    compose_symbolic_site_sparse(virtSpace_in, virtSpace_out, primary_ops)

Create a sparse matrix representation of a quantum operator that maps between input and output virtual spaces.

This function composes a sparse matrix representation where each non-zero entry contains a collection 
of site operators. It is the sparse implementation counterpart to the dense operator composition function.

# Arguments
- `virtSpace_in`: Input virtual space - array of possible input states
- `virtSpace_out`: Output virtual space - array of possible output states where each state contains
  composed operators
- `primary_ops`: Array of primary operators used in the composition

# Returns
- A sparse matrix of dimensions `(length(virtSpace_in), length(virtSpace_out))` where non-zero entries 
  contain vectors of `SiteOp` objects multiplied by appropriate factors.

# Note
The function tracks non-zero entries using explicit coordinate lists (COO format) and
groups operators that map between the same input and output states.
"""
function compose_symbolic_site_sparse(virtSpace_in, virtSpace_out, primary_ops)
    # Use a dictionary to accumulate entries
    # Key: (row, col), Value: Vector of operators
    entries = Dict{Tuple{Int,Int}, Vector{SiteOp}}()
    
    # For each output operator
    for (ivs, vs_out) in enumerate(virtSpace_out)
        # For each composed operator in this output operator
        for op in vs_out
            position = (op.in_idx, ivs)
            
            # Create or retrieve the vector at this position
            if !haskey(entries, position)
                entries[position] = SiteOp[]
            end
            
            # Add the operator
            push!(entries[position], primary_ops[op.op_idx] * op.factor)
        end
    end
    
    # Create a sparse matrix from the dictionary
    I = Int[]
    J = Int[]
    V = Vector{SiteOp}[]
    
    for ((i, j), ops) in entries
        push!(I, i)
        push!(J, j)
        push!(V, ops)
    end
    
    return sparse(I, J, V, length(virtSpace_in), length(virtSpace_out))
end

"""
    QNParser(phySpace::GradedSpace)

Convert a `GradedSpace` to an appropriate quantum number parser function.
Returns a function that can parse spin-up and spin-down occupation numbers into the appropriate quantum number representation.

# Arguments
- `phySpace::GradedSpace`: The graded space defining the symmetry type

# Returns
- A function that converts spin occupation numbers to quantum numbers
"""
QNParser(phySpace::GradedSpace; double_vSpace=false) = QNParser(TensorKit.type_repr(sectortype(phySpace)); double_vSpace=double_vSpace)

"""
    QNParser(symm::String)

Get the appropriate quantum number parser function based on the symmetry type string representation.

# Arguments
- `symm::String`: String representation of the symmetry type

# Returns
- A function that converts spin occupation numbers to quantum numbers

# Supported symmetry types
- "(FermionParity ⊠ Irrep[U₁]": Uses fermion number conservation
- "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])": Uses Z₂ ⊠ U₁ ⊠ SU₂ symmetry

# Errors
- Throws an error if the symmetry type is not supported
"""
function QNParser(symm::String; double_vSpace=false)
    if symm == "(FermionParity ⊠ Irrep[U₁])" || lowercase(symm) == "u1"
        return fNumberQNParser
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])" || lowercase(symm) == "u1u1"
        return U1U1QNParser
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])" || lowercase(symm) in ("su2", "u1su2")
        if double_vSpace
            return SU2doubleQNParser
        else
            return SU2QNParser
        end
    else
        error("Unsupported quantum number representation: $symm")
    end
end

"""
    fNumberQNParser(spinUp::Int, spinDown::Int)

Convert spin-up and spin-down occupation numbers to a quantum number with fermion number conservation.

# Arguments
- `spinUp::Int`: Number of spin-up electrons
- `spinDown::Int`: Number of spin-down electrons

# Returns
- A Tuple with (Z₂ fermionic parity, U₁ total particles) quantum numbers
"""
function fNumberQNParser(spinUp::Int, spinDown::Int)
    u1 = spinUp + spinDown
    f = abs(u1) % 2
    return (f, u1)
end


function U1U1QNParser(spinUp::Int, spinDown::Int)
    total_cnt = spinUp + spinDown
    f = abs(total_cnt) % 2
    total_spin = (spinUp - spinDown) // 2
    return (f, total_cnt, total_spin)
end

"""
    SU2QNParser(spinUp::Int, spinDown::Int)

Convert spin-up and spin-down occupation numbers to a quantum number with Z₂ ⊠ U₁ ⊠ SU₂  symmetry.

# Arguments
- `spinUp::Int`: Number of spin-up electrons
- `spinDown::Int`: Number of spin-down electrons

# Returns
- A Tuple with (Z₂ fermionic parity, SU₂ total spin, U₁ total particles)
"""
function SU2QNParser(spinUp::Int, spinDown::Int)
    u1 = spinUp + spinDown
    su2 = abs(spinUp - spinDown) / 2
    f = abs(u1) % 2
    return (f, u1, su2)
end

function SU2singleQNParser(qn::Int)
    # For SU(2) single quantum number, qn is the total particle count
    fZ = abs(qn) % 2
    su2 = fZ // 2
    return (fZ, qn, su2)
end

SU2doubleQNParser(qn1::Int, qn2::Int) = (SU2singleQNParser(qn1), SU2singleQNParser(qn2))

macro addSpinIteratorCapability(func)
    return quote
        function $(esc(func))(spins::AbstractVector{<:Integer})
            @assert length(spins) == 2 "Input must contain exactly two integers: [spinUp, spinDown]"
            return $(esc(func))(spins[1], spins[2])
        end
        
        function $(esc(func))(spins)
            return $(esc(func))(collect(spins))
        end
    end
end

# Add iterator methods to both functions
@addSpinIteratorCapability fNumberQNParser
@addSpinIteratorCapability U1U1QNParser
@addSpinIteratorCapability SU2QNParser
@addSpinIteratorCapability SU2doubleQNParser

"""
    count_ops(ops::Vector{String}) -> Tuple{Int, Int, Int, Int, Int}

Count the different types of operators in a vector of operator strings.

# Arguments
- `ops::Vector{String}`: A vector of operator strings, where each string can contain
  creation operators (`c↑`, `c↓`), annihilation operators (`a↑`, `a↓`), and identity operators (`I`).

# Returns
- `cUpCount::Int`: Number of creation operators with up spin.
- `cDownCount::Int`: Number of creation operators with down spin.
- `aUpCount::Int`: Number of annihilation operators with up spin.
- `aDownCount::Int`: Number of annihilation operators with down spin.
- `iCount::Int`: Number of identity operators.
"""

function count_ops(ops::Vector{String})
    cUpCount = 0
    cDownCount = 0
    aUpCount = 0
    aDownCount = 0
    iCount = 0
    
    for op in ops
        if op == "I"
            iCount += 1
            continue
        end
        chars = collect(op)
        
        # Process the string in pairs
        for i in 1:2:length(chars)
            if i+1 <= length(chars)
                action = chars[i]
                spin = chars[i+1]
                
                if action == 'a'  # Annihilation
                    if spin == '↑'
                        aUpCount += 1
                    elseif spin == '↓'
                        aDownCount += 1
                    end
                elseif action == 'c'  # Creation
                    if spin == '↑'
                        cUpCount += 1
                    elseif spin == '↓'
                        cDownCount += 1
                    end
                end
            end
        end
    end
    return cUpCount, cDownCount, aUpCount, aDownCount, iCount
end

"""
    symbolic_mpo_ops_counter(symbolic_mpo; print=true) -> Vector{Tuple{Int, Int, Int, Int, Int}}

Count the types of operators present in each site of a symbolic MPO representation.

# Arguments
- `symbolic_mpo`: A symbolic Matrix Product Operator representation of a Hamiltonian
- `print`: Boolean flag indicating whether to print the counts for each site (default: true)

# Returns
A vector of tuples, where each tuple contains the counts of each operator type at the corresponding site:
- Creation operator for spin-up (c↑)
- Creation operator for spin-down (c↓)
- Annihilation operator for spin-up (a↑)
- Annihilation operator for spin-down (a↓)
- Identity operator (I)
"""
function symbolic_mpo_ops_counter(symbolic_mpo; print=true)
    n_sites = length(symbolic_mpo)
    ops_counts = Vector{Tuple{Int, Int, Int, Int, Int}}(undef, n_sites)
    for i in 1:n_sites
        ops = [op.operator for ops in findnz(symbolic_mpo[i])[3] for op in ops]
        ops_counts[i] = count_ops(ops)
    end
    if print
        for i in 1:n_sites
            cUpCount, cDownCount, aUpCount, aDownCount, iCount = ops_counts[i]
            println("Site $i: c↑=$cUpCount, c↓=$cDownCount, a↑=$aUpCount, a↓=$aDownCount, I=$iCount")
        end
    end
    return ops_counts
end


function get_QN_mapping_and_vs_multiplicity(symbolic_mpo, virt_spaces_U1U1, qn_parser; verbose=1, symm="(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])", double_vSpace=false)
    if symm == "(FermionParity ⊠ Irrep[U₁])"
        error("get_QN_mapping_and_vs_multiplicity for (FermionParity ⊠ Irrep[U₁]) has not been implemented yet.")
        return get_QN_mapping_and_vs_multiplicity_non_spin_symmetric(virt_spaces_U1U1, qn_parser)
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])"
        return get_QN_mapping_and_vs_multiplicity_non_spin_symmetric(virt_spaces_U1U1, qn_parser)
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])"
        if double_vSpace
            return get_symmQN_mapping_and_vs_SU2_doubleVS(virt_spaces_U1U1, qn_parser)
        else
            return get_symmQN_mapping_and_vs_SU2(symbolic_mpo, virt_spaces_U1U1, qn_parser; verbose=verbose)
        end
    else
        error("Unsupported quantum number representation: $symm")
    end
end


function get_qn_vs_maps__qn_mult_counts_doubleVS(virt_space, qn_parser)
    """
    Get the quantum number mapping and multiplicity counts for a double virtual space.
    
    Parameters:
    - virt_space_U1U1: A virtual space with U1U1 quantum numbers
    - qn_parser: Function to parse quantum numbers
    - verbose: Verbosity level (default: 1)
    
    Returns:
    - A tuple containing the quantum number mapping and multiplicity counts
    """
    virt_space_qns = qn_parser.(first.(virt_space))
    single_qn_type = typeof(qn_parser(0,0)[1])
    col_to_qn_map = Dict{Int64, Vector{Tuple{single_qn_type, Int, Int}}}()
    qn_mult_cnt = DefaultDict{single_qn_type, Int64}(() -> 0) # Counts the muntiplicity in the MPO site to be constructed
    for (i_col, qn_doubleV) in enumerate(virt_space_qns)
        fZ, eCnt, totSpin = collect.(zip(qn_doubleV...))
        eCnt_fused = sum(eCnt)
        totSpin_fused = sum(totSpin)
        if totSpin_fused == 1 # different cases from (0,0,0); another for 020 where the operator multiplicity is 1 for (1,-1) and 2 for (-1,1), as there is no other case
            qn1 = (0, eCnt_fused, 0//1)
            qn2 = (0, eCnt_fused, 1//1)
            qn_mult_cnt[qn1] += 1
            qn_mult_cnt[qn2] += 1
            # The trivial space (0, 0, 0) must be distinguished when it is fused from two trivial spaces (op_mult=1)
            # or from (1, 1, 1//2) and (1, -1, 1//2) in the double virtual space (op_mul=2); each with a different behavior.
            # On the other hand, the (0, +/- 2, 0) spaces can be uniquely defined with a single op_mult. The fermionic
            # antisymmetry between the construction of (0, -2, 0) ((0, 2, 0)) from c1c2 (a1a2) and c2c1 (a2a1) is preserved by adding a negative sign
            # to their fusion tree
            op_mult_spin0 = eCnt_fused == 0 ? 2 : 1
            col_to_qn_map[i_col] = [(qn1, op_mult_spin0, qn_mult_cnt[qn1]), (qn2, 1, qn_mult_cnt[qn2])]
        elseif totSpin_fused == 1//2
            # We will need to distinguish the cases where the VS comes from the first (+-1, 0) -multiplicity 1- or second operator (0, +-1) -multiplicity 2-
            qn = (1, eCnt_fused, 1//2)
            qn_mult_cnt[qn] += 1
            mult = iszero(eCnt[2]) ? 1 : 2
            col_to_qn_map[i_col] = [(qn, mult, qn_mult_cnt[qn])]
        else
            fZ_fused = abs(eCnt_fused) % 2
            totSpin_fused = abs(totSpin_fused) // 1
            qn = (fZ_fused, eCnt_fused, totSpin_fused)
            qn_mult_cnt[qn] += 1
            col_to_qn_map[i_col] = [(qn, 1, qn_mult_cnt[qn])]
        end
    end

    return col_to_qn_map, convert(Dict, qn_mult_cnt)
end

function get_symmQN_mapping_and_vs_SU2_doubleVS(virt_spaces_U1U1, qn_parser)
    qn_vs_maps__qn_mult_counts = [get_qn_vs_maps__qn_mult_counts_doubleVS(vsU1U1, qn_parser) for vsU1U1 in virt_spaces_U1U1]
    qn_vs_maps, qn_mult_counts = zip(qn_vs_maps__qn_mult_counts...)
    return qn_vs_maps, qn_mult_counts
end

function get_qn_vs_maps__qn_mult_counts(virt_space_U1U1, qn_parser)
    """
    Get the quantum number mapping and multiplicity counts for a single virtual space.
    
    Parameters:
    - virt_space_U1U1: A virtual space with U1U1 quantum numbers
    - qn_parser: Function to parse quantum numbers
    - verbose: Verbosity level (default: 1)
    
    Returns:
    - A tuple containing the quantum number mapping and multiplicity counts
    """
    virt_space_U1U1_qns = qn_parser.(first.(virt_space_U1U1))
    qn_type = typeof(qn_parser(0,0))
    col_to_qn_map = Dict{Int64, Tuple{qn_type, Int}}()
    qn_mult_cnt = DefaultDict{qn_type, Int64}(() -> 0)
    for (i_col, qn) in enumerate(virt_space_U1U1_qns)
        qn_mult_cnt[qn] += 1
        col_to_qn_map[i_col] = (qn, qn_mult_cnt[qn])
    end

    return col_to_qn_map, convert(Dict, qn_mult_cnt)
end

function get_QN_mapping_and_vs_multiplicity_non_spin_symmetric(virt_spaces_U1U1, qn_parser)
    qn_vs_maps__qn_mult_counts = [get_qn_vs_maps__qn_mult_counts(vsU1U1, qn_parser) for vsU1U1 in virt_spaces_U1U1]
    qn_vs_maps, qn_mult_counts = zip(qn_vs_maps__qn_mult_counts...)
    return qn_vs_maps, qn_mult_counts
end


flipOp(op::String) = replace(op, "↑" => "↓", "↓" => "↑")

function find_flipped_ops(ops_dict::Dict, ops::Vector{SiteOp}, row::Int64, col::Int64, vs_leftQNs::Vector{Tuple{Int64, Int64}}, vs_rightQNs::Vector{Tuple{Int64, Int64}}; row_qn_map::Union{Nothing, Dict{Int64, @NamedTuple{qn::Tuple{Int64, Int64, Float64}, mult::Int64, m::Rational{Int64}}}}=nothing)
    coefs = sort([op.coefficient for op in ops])
    flippedOpsSet = Set(flipOp(op.operator) for op in ops)
    OpLeftQN = vs_leftQNs[row]
    OpRightQN = vs_rightQNs[col]

    results = []
    for (key, (this_row, this_col, this_ops)) in ops_dict
        thisOpsSet = Set(op.operator for op in this_ops)
        thisCoefs = sort([op.coefficient for op in this_ops])
        # this_ops_leftQNs = vs_leftQNs[this_row]
        # this_ops_rightQNs = vs_rightQNs[this_col]
        
        # # TODO: approx of coeff isapprox(coefs, thisCoefs) if needed
        # if thisOpsSet == flippedOpsSet && thisCoefs == coefs &&
        #    ((this_ops_leftQNs == this_ops_rightQNs == OpLeftQN == OpRightQN) || (this_ops_leftQNs == (OpLeftQN[2], OpLeftQN[1]) &&
        #    this_ops_rightQNs == (OpRightQN[2], OpRightQN[1]))) && (row == this_row || col == this_col || (row_qn_map !== nothing && row_qn_map[row].qn == row_qn_map[this_row].qn && row_qn_map[row].mult == row_qn_map[this_row].mult))
        #     push!(results, key)
        # end
        if row_qn_map[row].qn == row_qn_map[this_row].qn &&
            row_qn_map[row].mult == row_qn_map[this_row].mult &&
            # (row_qn_map[row].qn[3] == row_qn_map[this_row].qn[3] == 0 || (row_qn_map[row].qn[3] > 0 && row_qn_map[row].m + row_qn_map[this_row].m == 0)) && # Assumes SU2 symmetry and it position 3 in the QN. Should be generalized, where the magnetic projection is checked according to the symmetry type (not sure if needed anymore. checking for all projections?)
            row_qn_map[row].m == -row_qn_map[this_row].m && # Assumes SU2 symmetry and it position 3 in the QN. Should be generalized, where the magnetic projection is checked according to the symmetry type (not sure if needed anymore. checking for all projections?)
            # col== this_col &&
            thisOpsSet == flippedOpsSet &&
            isapprox(thisCoefs, coefs, atol=1e-8)
            push!(results, key)
        end

    end

    if length(results) > 1
        results = filter(x -> ops_dict[x][2] == col, results)
        @assert length(results) == 1 "Even after filtering multiple results, could not find exactly one match for flipped operators at ($row, $col), found $(length(results)) matches: $(results)"
    # else
    #     @assert length(results) == 1 "Expected exactly one match for flipped operators at ($row, $col), found $(length(results)) matches: $(results)"
    end
    # TODO: When throwing an error, mention which was the flipped counterpart operator that was missing

    # if length(results) > 1
    #     filtered_res = []
    #     for idx in results
    #         res = dict[idx]
    #         if res[1] == row #|| res[2] == col
    #             push!(filtered_res, idx)
    #         end
    #     end
    #     if length(filtered_res) == 1
    #         results = filtered_res
    #     else
    #         println("$(length(filtered_res)) matches found after filtering for row $row and col $col:")
    #         for idx in filtered_res
    #             println("  - ", idx, " => ", dict[idx])
    #         end
    #         error("Multiple matches found for flipped ops with row $row and col $col")
    #     end
    # end


    return results
end

function match_mpo_ops(symb_mpo_site, vs_left, vs_right; verbose=false, row_qn_map::Union{Nothing, Dict{Int64, @NamedTuple{qn::Tuple{Int64, Int64, Float64}, mult::Int64, m::Rational{Int64}}}}=nothing)
    vs_leftQNs = first.(vs_left)
    vs_rightQNs = first.(vs_right)
    matches = []
    nz_iter = zip(findnz(symb_mpo_site)...)
    ops_dict = Dict(i => val for (i, val) in enumerate(nz_iter))

    # Create a list of keys to process
    keys_to_process = collect(keys(ops_dict))
    verbose && println("Total keys to process: ", length(keys_to_process))

    isnothing(row_qn_map) && println("No row mapping provided, using default row indices.")
    find_flipped_ops_thisSite(ops_dict, ops, row, col) = find_flipped_ops(ops_dict, ops, row, col, vs_leftQNs, vs_rightQNs, row_qn_map=row_qn_map)

    while !isempty(keys_to_process)
        i = popfirst!(keys_to_process)
        
        # TODO: Remove this check when stable
        # Skip if this key was already processed (removed)
        if !haskey(ops_dict, i)
            error("Key $i already processed, shouldn't get here.")
        end
        
        verbose && println("Processing key: ", i, " with val", ops_dict[i], " of ", length(keys_to_process), " remaining keys")
        
        val = pop!(ops_dict, i)
        row, col, ops = val
        vsL_qn = vs_leftQNs[row]
        vsR_qn = vs_rightQNs[col]
        
        # Try to find flipped ops among remaining items
        flipped_ops_idx = find_flipped_ops_thisSite(ops_dict, ops, row, col)
        verbose && println("Found flipped ops for $(val): ", [ops_dict[j] for j in flipped_ops_idx])
        if isempty(flipped_ops_idx) # || flipped_ops_idx == [i] # i was removed from ops_dict
            flipped_ops_idx = find_flipped_ops_thisSite(Dict(i=>val), ops, row, col)
            @assert only(flipped_ops_idx) == i "Expected only one flipped operator for ops at ($row, $col), but found multiple: $(flipped_ops_idx). This suggests an imbalance in the Hamiltonian terms.\nThis op: $(val)\nvsL_qn: $(vsL_qn)\nvsR_qn: $(vsR_qn)"
            match = (nMatches=1, rows=(row,), vsL_qn=(vsL_qn,), cols=(col,), vsR_qn=(vsR_qn,), ops=(ops,))
            push!(matches, match)
            delete!(ops_dict, i)  # Remove the self matched key from the dictionary

            # if length(ops)==1 && ops[1].operator != "I"
            #     # If no flipped ops found (including itself), something is wrong
            #     println(ops_dict)
            #     error("No flipped operators found for ops at ($row, $col). This suggests an imbalance in the Hamiltonian terms.\nThis op: $(val)")
            # else
            #     # If the only operator is "I", we can skip it
            #     match = (nMatches=1, rows=(row,), vsL_qn=(vsL_qn,), cols=(col,), vsR_qn=(vsR_qn,), ops=(ops,))
            #     push!(matches, match)
            #     delete!(ops_dict, i)  # Remove the self matched key from the dictionary
            # end
        elseif length(flipped_ops_idx) > 1
            ms=[ops_dict[k] for k in flipped_ops_idx]
            matches_info = [((m[1],vs_left[m[1]][1]), (m[2],vs_right[m[2]][1]), m[3]) for m in ms]
            error("Multiple flipped operators found for ops at ($row, $col) with index $i. This suggests an imbalance in the Hamiltonian terms.\nThis op: $(val)\nvsL_qn: $(vsL_qn)\nvsR_qn: $(vsR_qn)\nMatches: $(matches_info)")
        #     # If multiple flipped ops found, this is unexpected

        #     similar_ops = collect(Set([ops_dict[idx] for (row_f, col_f, ops_f) in ms for idx in find_flipped_ops_thisSite(ops_dict, ops_f, row_f, col_f)]))
        #     if length(similar_ops) + 1 == length(flipped_ops_idx)
        #         nMatches = length(flipped_ops_idx) * 2
        #         # Use list comprehensions for single-step extraction of all components
        #         all_rows = [row, [m[1] for m in ms]..., [s[1] for s in similar_ops]...]
        #         all_cols = [col, [m[2] for m in ms]..., [s[2] for s in similar_ops]...]
        #         all_ops = [ops..., [m[3] for m in ms]..., [s[3] for s in similar_ops]...]
        #         all_vsL_qn = [vsL_qn, [vs_left[m[1]][1] for m in ms]..., [vs_left[s[1]][1] for s in similar_ops]...]
        #         all_vsR_qn = [vsR_qn, [vs_right[m[2]][1] for m in ms]..., [vs_right[s[2]][1] for s in similar_ops]...]
        #         match = (nMatches=nMatches, rows=all_rows, vsL_qn=all_vsL_qn, cols=all_cols, vsR_qn=all_vsR_qn, ops=all_ops)
        #         push!(matches, match)
        #         # Remove all matched keys from the dictionary
        #         for idx in flipped_ops_idx
        #             delete!(ops_dict, idx)
        #         end
        #         for idx in similar_ops
        #             delete!(ops_dict, idx)
        #         end
        #         filter!(k -> !(k in flipped_ops_idx), keys_to_process)
        #         filter!(k -> !(k in similar_ops), keys_to_process)


        #     else
        #         similar_ops_info = [((s[1],vs_left[s[1]][1]), (s[2],vs_right[s[2]][1]), s[3]) for s in similar_ops]
        #         error("Mismatch in number of similar ops found: $(length(similar_ops)+1) vs $(length(flipped_ops_idx)). This suggests an imbalance in the Hamiltonian terms.\nThis op: $(val)\nvsL_qn: $(vsL_qn)\nvsR_qn: $(vsR_qn)\nMatches: $(matches_info)\nSimilar ops: $(similar_ops_info)")
        #     end

        #     # error("Multiple flipped operators found for ops at ($row, $col) with index $i. This suggests an imbalance in the Hamiltonian terms.\nThis op: $(val)\nMatches: $(ms)")

        else
            flipped_ops_idx = flipped_ops_idx[1]  # Get the single index of the flipped ops
            # Remove the matched key from the list to process
            filter!(k -> k != flipped_ops_idx, keys_to_process)
            
            rowF, colF, opsF = pop!(ops_dict, flipped_ops_idx)
            vsL_qnF = vs_left[rowF][1]
            vsR_qnF = vs_right[colF][1]
            match = (nMatches=2, rows=(row, rowF), vsL_qn=(vsL_qn, vsL_qnF), cols=(col, colF), vsR_qn=(vsR_qn, vsR_qnF), ops=(ops, opsF))
            push!(matches, match)
        end

        

    end
    sort!(matches, by=m -> m.cols[1])
    return matches
end

function group_col_matches(matches)
    cols_matches = DefaultDict{Int64, Set{Int64}}(() -> Set{Int64}())
    for match in matches
        for col in match.cols
            for col_ in match.cols
                push!(cols_matches[col], col_)
            end
        end
    end
    col_grps = Set(values(cols_matches))
    @assert all(length.(col_grps) .<= 3)
    col_grps = [Tuple(sort(collect(grp))) for grp in col_grps]
    sort!(col_grps, by = first)
    return Tuple(col_grps)
end

function get_new_symbolic_spin_symmetric_col_mapping(col_grps, vs_right; verbose=false)
    vs_rightQNs = first.(vs_right)
    # Map the column indexes (associated with QNs) to group indexes including magnetic_projection.
    # Convention for ordering two given column indexes in a group:
    # - First by the count of spin down electrons. Note that due to symmetry, counting spin down electrons is redundant.
    # - If the counts are equal, order by the column index itself.
    # This ensures that the groups are ordered consistently and uniquely.
    col_to_group_map = Dict{Int64, Tuple{Int64, Int64}}()
    new_col_idx = 1
    for (igrp, col_grp) in enumerate(col_grps)
        verbose && println("Processing group ", igrp, " with columns: ", col_grp)
        if length(col_grp) == 1
            # If the group has only one column, it is not degenerate
            col_to_group_map[col_grp[1]] = (new_col_idx, 1)
            verbose && println("Single column: ", col_grp[1], " -> ", col_to_group_map[col_grp[1]])
        else
            col1, col2 = col_grp
            vs1_qn = vs_rightQNs[col1]
            vs2_qn = vs_rightQNs[col2]
            verbose && println("vs1_qn: ", vs1_qn, " vs2_qn: ", vs2_qn)
            
            if vs1_qn[1] > vs2_qn[1]
                col_to_group_map[col1] = (new_col_idx, 1) 
                col_to_group_map[col2] = (new_col_idx, 2)
            elseif vs1_qn[1] < vs2_qn[1]
                col_to_group_map[col1] = (new_col_idx, 2) 
                col_to_group_map[col2] = (new_col_idx, 1)
            else  # Same number of spin up electrons (and thus same number of spin down electrons)
                if col1 <= col2
                    col_to_group_map[col1] = (new_col_idx, 1) 
                    col_to_group_map[col2] = (new_col_idx, 2)
                else
                    col_to_group_map[col1] = (new_col_idx, 2)
                    col_to_group_map[col2] = (new_col_idx, 1)
                end
            end
            verbose && println("col1: ", col1, " -> ", col_to_group_map[col1])
            verbose && println("col2: ", col2, " -> ", col_to_group_map[col2])
        end
        new_col_idx += 1
    end

    if verbose
        sorted_cols_mapping = sort(collect(col_to_group_map), by = x -> (x[2][1], x[2][2]))  # Sort by group index
        vs_rightQNs = first.(vs_right)
        for (col, (group_idx, deg_idx)) in sorted_cols_mapping
            println("Group ", group_idx, " (", deg_idx, ") <- Col ", col, " with QN ", vs_rightQNs[col])
        end
    end

    return col_to_group_map
end

function group_vs_for_spin_symmetry(symb_mpo_site, vs_left, vs_right; row_qn_map::Union{Nothing, Dict{Int64, @NamedTuple{qn::Tuple{Int64, Int64, Float64}, mult::Int64, m::Rational{Int64}}}}=nothing, verbose=0)
    # Tested also for cases with virtual spaces with multiple spinUp and spinDown quantum numbers (no tolerance for OpSum construction)
    matches = match_mpo_ops(symb_mpo_site, vs_left, vs_right; verbose=verbose>1, row_qn_map=row_qn_map)
    col_grps = group_col_matches(matches)
    verbose > 0 && println("Grouped ", length(vs_right), " trivial Virtual Spaces to ", length(col_grps), " matches by spin symmetry.")
    return col_grps
end

function get_symmQN_mapping_and_vs_from_col_grps(col_grps, vsU1U1_right_, qn_parser; verbose=false)
    vsU1U1_right = first.(vsU1U1_right_)
    # Map the column indexes (associated with QNs) to group indexes including degeneracies.
    # Convention for ordering two given column indexes in a group:
    # - First by the count of spin down electrons. Note that due to symmetry, counting spin down electrons is redundant.
    # - If the counts are equal, order by the column index itself.
    # This ensures that the groups are ordered consistently and uniquely.
    col_to_qn_map = Dict{Int64, Any}()
    qn_mult_cnt = DefaultDict{typeof(qn_parser(0,0)), Int64}(() -> 0) # Quantum number multiplicity count
    for (igrp, col_grp) in enumerate(col_grps)
        verbose && println("Processing group ", igrp, " with columns: ", col_grp)
        if length(col_grp) == 1
            # If the group has only one column, it is not degenerate
            spinUp_cnt, spinDown_cnt = vsU1U1_right[col_grp[1]]
            symmQN = qn_parser((spinUp_cnt, spinDown_cnt))
            qn_mult_cnt[symmQN] += 1
            m = (spinUp_cnt - spinDown_cnt) // 2
            col_to_qn_map[col_grp[1]] = (symmQN, qn_mult_cnt[symmQN], m)
            verbose && println("Single column: ", col_grp[1], " -> ", col_to_qn_map[col_grp[1]])
        else
            col1, col2 = col_grp
            vs1_U1U1qn = vsU1U1_right[col1]
            m1 = (vs1_U1U1qn[1] - vs1_U1U1qn[2]) // 2
            vs2_U1U1qn = vsU1U1_right[col2]
            m2 = (vs2_U1U1qn[1] - vs2_U1U1qn[2]) // 2
            symmQN = qn_parser(vsU1U1_right[col1]) # Same for both columns in the group
            qn_mult_cnt[symmQN] += 1
            mult = qn_mult_cnt[symmQN]
            col_to_qn_map[col1] = (symmQN, mult, m1)
            col_to_qn_map[col2] = (symmQN, mult, m2)
            
            verbose && println("col1: ", col1, " -> ", col_to_qn_map[col1])
            verbose && println("col2: ", col2, " -> ", col_to_qn_map[col2])
        end
    end

    if verbose
        sorted_cols_mapping = sort(collect(col_to_qn_map), by = x -> (x[2][1], x[2][2], x[2][3]))  # Sort by qn index
        for (col, (qn, mult, m)) in sorted_cols_mapping
            println("QN (Z₂ ⊠ U₁ ⊠ SU₂):", qn, ", multiplicity ", mult, ", magnetic_projection", m, " <- Col ", col, " with U1U1qn ", vsU1U1_right[col])
        end
    end

    return col_to_qn_map, qn_mult_cnt
end

function get_symmQN_mapping_and_vs_SU2(symbolic_mpo, virt_spaces_U1U1, qn_parser; verbose=1)

    trivial_map = (qn=qn_parser(0,0), mult=1, m=0//1)
    qn_vs_maps = Vector{Dict{Int64, typeof(trivial_map)}}(undef, length(symbolic_mpo)+1)
    qn_vs_maps[1] = Dict(1 => trivial_map)
    qn_mult_counts = Vector{Dict{typeof(qn_parser(0,0)), Int64}}(undef, length(symbolic_mpo)+1)
    qn_mult_counts[1] = Dict(qn_parser(0,0) => 1)  # Initialize with trivial quantum number

    for (isite, symb_mpo_site) in enumerate(symbolic_mpo)
        vsU1U1_left = virt_spaces_U1U1[isite]
        vsU1U1_right = virt_spaces_U1U1[isite+1]
        col_grps = group_vs_for_spin_symmetry(symb_mpo_site, vsU1U1_left, vsU1U1_right; row_qn_map=qn_vs_maps[isite], verbose=verbose)
        col_to_qn_map, qn_mult_cnt = get_symmQN_mapping_and_vs_from_col_grps(col_grps, vsU1U1_right, qn_parser; verbose=verbose>1)
        qn_vs_maps[isite+1] = col_to_qn_map  # Store the mapping for this site
        qn_mult_counts[isite+1] = qn_mult_cnt  # Store the multiplicity counts for this site
    end


    return qn_vs_maps, qn_mult_counts
end

function construct_empty_mpo_site(phySpace::GradedSpace, left_qn_mult::Dict{QN, Int64}, right_qn_mult::Dict{QN, Int64}; dataType::DataType=Float64) where QN
    """
    Construct an empty MPO site with the given quantum number multiplicities.
    
    Parameters:
    - phySpace: Physical space (GradedSpace) for TensorKit tensors
    - left_qn_mult: Multiplicity count for each left quantum number
    - right_qn_mult: Multiplicity count for each right quantum number
    - dataType: Data type for tensor elements
    
    Returns:
    - An empty MPO site represented as a TensorMap
    """
    qn_space = typeof(phySpace)
    virtSpace_left = reduce(⊕, [qn_space(qn => cnt) for (qn, cnt) in left_qn_mult])
    virtSpace_right = reduce(⊕, [qn_space(qn => cnt) for (qn, cnt) in right_qn_mult])

    
    # Create a TensorMap for this site
    return zeros(dataType, virtSpace_left ⊗ phySpace ← virtSpace_right ⊗ phySpace)
end

function get_symbolic_operators_iter(symb_mpo_site, vs_map_left, vs_map_right)

    ops_set = Set()
    for (row, col, ops) in zip(findnz(symb_mpo_site)...)
        left_vs_info = vs_map_left[row]
        right_vs_info = vs_map_right[col]
        for op in ops
            # symm_op_string = op.operator == "I" ? "I" : join([collect(op.operator)[i] for i in 1:2:length(op.operator)]) # Remove the spin labels from the operator string

            # Create a unique identifier for the operator based on its components
            # Should we check if the coefficient is approximately equal to another op with all the same other components?
            # TODO: use dict haskey where value is coeff and compare coeffs
            op_id = (left_vs_info.qn, left_vs_info.mult, left_vs_info.m, right_vs_info.qn, right_vs_info.mult, right_vs_info.m, op.coefficient, op.operator)
            push!(ops_set, op_id)
        end
    end
    return ops_set
end

function GetCrAnLocalOpsU1U1(phySpace; dataType::DataType=Float64, spin_symm::Bool=false)
    """ Generate local operators for the Hamiltonian """
    # TOCHECK: Should they be defined as the adjoint conjugate? Is that why SpinUp and SpinDown are reversed?
    # Creation and annilihation operator for spin-1/2 multiplet (spinUp and spinDown)
    # Notes:
    # - The orbital basis states are in the following order {|∅>, |↑>, |↓>, |↑↓>}
    # - The filling antisymmetric convention is: ↑↓ is positive; and ↓↑ is negative.
    # - Fillings are done from the left: a↑†a↓†|∅> = |↑↓> ; while a↓†a↑†|∅> = |↓↑> = -|↑↓>
    # - Filling a spinUp electron will have a positive sign in the representation array of the creation operator, while filling a spinDown electron will have a negative sign.
    # - The creation/annihilation operators are constructed by iterating over the fusion trees of the complete space (virt⊗phy ⊗ virt⊗phy) that increase/decrease the U1 total count by 1,
    #   then building a "one-hot" tensormap and converting it to its array representation. The fusiontree intersection value is defined by the coefficient that would result in a +1 value in the array representation
    #   for creating/annihilating a spinUp electron (from |∅> to |↑> or from |↑> to |∅>) in the physical space. This coefficient would be the inverse of the Glebsch Gordan coefficient for the fusion tree intersection.
    
    # 13 virtual states:
    # 1:  (0, 0, 0)        |∅>
    # 2:  (0, 0, 1)        |↑>-|↓>
    # 3:  (0, 0, -1)      -|↑>+|↓>
    # 4:  (1, 1, 1/2)      |↑>
    # 5:  (1, 1, -1/2)     |↓>
    # 6:  (1, -1, 1/2)    -|↓>
    # 7:  (0, 2, 0)        |↑↓>
    # 8:  (1, -1, -1/2)   -|↑>
    # 9:  (0, -2, 0)      -|↑↓>
    # 10: (0, 2, 1)        |↑↑>
    # 11: (0, 2, -1)       |↓↓>
    # 12: (0, -2, 1)      -|↓↓>
    # 13: (0, -2, -1)     -|↑↑>

    auxVecSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])]((0, 0, 0)=>1, (0, 0, 1)=>1, (0, 0, -1)=>1, (1, 1, 1/2)=>1, (1, 1, -1/2)=>1,
                                                                (1, -1, 1/2)=>1, (0, 2, 0)=>1, (1, -1, -1/2)=>1, (0, -2, 0)=>1,
                                                                (0, 2, 1)=>1, (0, 2, -1)=>1, (0, -2, 1)=>1, (0, -2, -1)=>1)
    
    # Virtual states creation pairings
    # (Incoming, outgoing, factor).
    # With negative virtual factors
    # Non-hole regime: Negative only from spinDown to double occupancy. Even |↑> ⟺ |↑↑> is positive.
    # Hole regime: Virtual spaces are opposite. E.g. -|↓> ⟺ -|↓↓> is negative; -|↑> ⟺ -|↑↓> is positive (double negative); -|↓> ⟺ |∅> is negative.
    # Positive and negative virtual spaces are perpendicular. E.g. -|↑>+|↓> = |↓>+-|↑> ; and -|↑>+|↓> ⟺ |↓> is positive
    # No fermionic anticommutation in the virtual space so |↑> (4) ⟺ |↑↓> (7) is positive
    # -|↓> (6) ⟺ -|↑↓> (9) is negative, -|↑> (8) ⟺ -|↑↓> (9) is also negative because there is no fermionic anticommutation
    vCrUpPairings = [(1,4,1), (3,5,-1), (4,10,1), (5,7,1), (6,2,1), (8,1,-1), (9,6,-1), (13,8,-1)]
    vCrDownPairings = [(1,5,1), (2,4,-1), (4,7,1), (5,11,1), (6,1,-1), (8,3,1), (9,8,-1), (12,6,-1)]
    # vCrDownPairings = [(1,5,1), (2,4,1), (4,7,1), (5,11,1), (6,1,1), (8,3,1), (9,8,-1), (12,6,-1)]
    vAnUpPairings = [(1,8,-1), (2,6,1), (4,1,1), (5,3,-1), (6,9,-1), (7,5,1), (8,13,-1), (10,4,1)]
    vAnDownPairings = [(1,6,-1), (3,8,1), (4,2,-1), (5,1,1), (6,12,-1), (7,4,1), (8,9,-1), (11,5,1)]
    # vAnDownPairings = [(1,6,1), (3,8,1), (4,2,1), (5,1,1), (6,12,-1), (7,4,1), (8,9,-1), (11,5,1)]

    # Physical creation pairings
    pCrUpPairings = [(1,2,1), (3,4,1)]
    pCrDownPairings = [(1,3,1), (2,4,-1)]
    pAnUpPairings = [(2,1,1), (4,3,1)]
    pAnDownPairings = [(3,1,1), (4,2,-1)]

    # Creation operator    
    crUpMat = zeros(dataType, TensorKit.dim(auxVecSpace), 4, TensorKit.dim(auxVecSpace), 4)
    for ((vIn, vOut, vFactor), (pIn, pOut, pFactor)) in Iterators.product(vAnUpPairings, pCrUpPairings)
        crUpMat[vOut, pOut, vIn, pIn] = vFactor * pFactor
    end
    crDownMat = zeros(dataType, TensorKit.dim(auxVecSpace), 4, TensorKit.dim(auxVecSpace), 4)
    for ((vIn, vOut, vFactor), (pIn, pOut, pFactor)) in Iterators.product(vAnDownPairings, pCrDownPairings)
        crDownMat[vOut, pOut, vIn, pIn] = vFactor * pFactor
    end

    # Annihilation operator
    anUpMat = zeros(dataType, TensorKit.dim(auxVecSpace), 4, TensorKit.dim(auxVecSpace), 4)
    for ((vIn, vOut, vFactor), (pIn, pOut, pFactor)) in Iterators.product(vCrUpPairings, pAnUpPairings)
        anUpMat[vOut, pOut, vIn, pIn] = vFactor * pFactor
    end
    anDownMat = zeros(dataType, TensorKit.dim(auxVecSpace), 4, TensorKit.dim(auxVecSpace), 4)
    for ((vIn, vOut, vFactor), (pIn, pOut, pFactor)) in Iterators.product(vCrDownPairings, pAnDownPairings)
        anDownMat[vOut, pOut, vIn, pIn] = vFactor * pFactor
    end

    idOp = localIdOp(phySpace, auxVecSpace; dataType=dataType)
    if spin_symm
        # Symmetric operators
        crMat = crUpMat + crDownMat
        anMat = anUpMat + anDownMat
        crOp = TensorMap(crMat, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
        anOp = TensorMap(anMat, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
        local_ops = LocalOps(crOp, anOp, idOp)
    else
        crUpOp = TensorMap(crUpMat, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
        crDownOp = TensorMap(crDownMat, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
        anUpOp = TensorMap(anUpMat, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
        anDownOp = TensorMap(anDownMat, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)

        local_ops = LocalOps(crUpOp, crDownOp, anUpOp, anDownOp, idOp)
    end
    
    return local_ops
end

function GetCrAnLocalOpsU1SU2(phySpace; dataType::DataType=Float64)
    """ Generate local operators for the Hamiltonian """
    # TOCHECK: Should they be defined as the adjoint conjugate? Is that why SpinUp and SpinDown are reversed?
    # Creation and annilihation operator for spin-1/2 multiplet (spinUp and spinDown)
    # Notes:
    # - The orbital basis states are in the following order {|∅>, |↑>, |↓>, |↑↓>}
    # - The filling antisymmetric convention is: ↑↓ is positive; and ↓↑ is negative.
    # - Fillings are done from the left: a↑†a↓†|∅> = |↑↓> ; while a↓†a↑†|∅> = |↓↑> = -|↑↓>
    # - Filling a spinUp electron will have a positive sign in the representation array of the creation operator, while filling a spinDown electron will have a negative sign.
    # - The creation/annihilation operators are constructed by iterating over the fusion trees of the complete space (virt⊗phy ⊗ virt⊗phy) that increase/decrease the U1 total count by 1,
    #   then building a "one-hot" tensormap and converting it to its array representation. The fusiontree intersection value is defined by the coefficient that would result in a +1 value in the array representation
    #   for creating/annihilating a spinUp electron (from |∅> to |↑> or from |↑> to |∅>) in the physical space. This coefficient would be the inverse of the Glebsch Gordan coefficient for the fusion tree intersection.
    
    # 16 virtual states:
    # 1: (0, 0, 0)      |∅>
    # 2: (0, 0, 1) m=1  |↑>-|↓>
    # 3: (0, 0, 1) m=0  [(|↑>-|↓>) + (-|↑>+|↓>)] / √2 =? (|↑↓> + -|↑↓>) / √2 (spin pointing in the +X direction in xy-plane)
    # 4: (0, 0, 1) m=-1 -|↑>+|↓>
    # 5: (1, 1, 1/2)    |↑>
    # 6: (1, 1, 1/2)    |↓>
    # 7: (1, -1, 1/2)   -|↓>
    # 8: (1, -1, 1/2)   -|↑>
    # 9: (0, 2, 0)      |↑↓>
    # 10: (0, -2, 0)     -|↑↓>
    # 11: (0, 2, 1)     |↑↑> m=+1
    # 12: (0, 2, 1)     (|↑↑> + |↓↓>) / √2 m=0 (spin pointing in the +Y direction in xy-plane)
    # 13: (0, 2, 1)     |↓↓> m=-1
    # 14: (0, -2, 1)    -|↓↓> m=+1
    # 15: (0, -2, 1)    (-|↑↑> + -|↓↓>) / √2 m=0 (spin pointing in the -Y direction in xy-plane?)
    # 16: (0, -2, 1)    -|↑↑> m=-1
    # auxVecSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0)=>2, (0, 0, 1)=>1, (1, 1, 1/2)=>1, (1, -1, 1/2)=>1, (0, 2, 0)=>1, (0, -2, 0)=>1, (0, 2, 1)=>1, (0, -2, 1)=>1)
    
    # New Convention + 3rd multiplicity on (0,0,0) to even the probabilities when constracting on the same site
    auxVecSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0)=>2, (0, 0, 1)=>1, (1, 1, 1/2)=>2, (1, -1, 1/2)=>2, (0, 2, 0)=>1, (0, -2, 0)=>1, (0, 2, 1)=>1, (0, -2, 1)=>1)

    ftree_type = FusionTree{sectortype(typeof(phySpace))}

    # CREATION OPERATOR

    # New Convention + 2 multiplicities for (1,+-1,1//2)
    # No additional multiplicities con contraction logics
    crOp_ftree_nzdata = [
        ((((1, -1, 0.5), (1, 1, 0.5)), (0, 0, 0.0), (false, false), ()), (((0, 0, 0.0), (0, 0, 0.0)), (0, 0, 0.0), (false, false), ()), [sqrt(2) 0; 0 1])
        ((((0, -2, 0.0), (0, 2, 0.0)), (0, 0, 0.0), (false, false), ()), (((1, -1, 0.5), (1, 1, 0.5)), (0, 0, 0.0), (false, false), ()), [0 -1])
        ((((1, -1, 0.5), (1, 1, 0.5)), (0, 0, 1.0), (false, false), ()), (((0, 0, 1.0), (0, 0, 0.0)), (0, 0, 1.0), (false, false), ()), [0; 1.0])
        ((((0, -2, 1.0), (0, 2, 0.0)), (0, 0, 1.0), (false, false), ()), (((1, -1, 0.5), (1, 1, 0.5)), (0, 0, 1.0), (false, false), ()), [0 -1.0])
        ((((0, 0, 0.0), (1, 1, 0.5)), (1, 1, 0.5), (false, false), ()), (((1, 1, 0.5), (0, 0, 0.0)), (1, 1, 0.5), (false, false), ()), [1.0 0; 0 1/sqrt(2)])
        ((((0, 0, 1.0), (1, 1, 0.5)), (1, 1, 0.5), (false, false), ()), (((1, 1, 0.5), (0, 0, 0.0)), (1, 1, 0.5), (false, false), ()), [0 sqrt(3/2)])
        ((((1, -1, 0.5), (0, 2, 0.0)), (1, 1, 0.5), (false, false), ()), (((0, 0, 0.0), (1, 1, 0.5)), (1, 1, 0.5), (false, false), ()), [-1.0 0; 0 -1/sqrt(2)])
        ((((1, -1, 0.5), (0, 2, 0.0)), (1, 1, 0.5), (false, false), ()), (((0, 0, 1.0), (1, 1, 0.5)), (1, 1, 0.5), (false, false), ()), [0; sqrt(3/2)])
        ((((0, -2, 0.0), (1, 1, 0.5)), (1, -1, 0.5), (false, false), ()), (((1, -1, 0.5), (0, 0, 0.0)), (1, -1, 0.5), (false, false), ()), [0 -1/sqrt(2)])
        ((((0, -2, 1.0), (1, 1, 0.5)), (1, -1, 0.5), (false, false), ()), (((1, -1, 0.5), (0, 0, 0.0)), (1, -1, 0.5), (false, false), ()), [0 sqrt(3/2)])
        ((((1, 1, 0.5), (1, 1, 0.5)), (0, 2, 0.0), (false, false), ()), (((0, 2, 0.0), (0, 0, 0.0)), (0, 2, 0.0), (false, false), ()), [0; -1])
        ((((0, 0, 0.0), (0, 2, 0.0)), (0, 2, 0.0), (false, false), ()), (((1, 1, 0.5), (1, 1, 0.5)), (0, 2, 0.0), (false, false), ()), [sqrt(2) 0; 0 1])
        ((((1, 1, 0.5), (1, 1, 0.5)), (0, 2, 1.0), (false, false), ()), (((0, 2, 1.0), (0, 0, 0.0)), (0, 2, 1.0), (false, false), ()), [0; 1.0])
        ((((0, 0, 1.0), (0, 2, 0.0)), (0, 2, 1.0), (false, false), ()), (((1, 1, 0.5), (1, 1, 0.5)), (0, 2, 1.0), (false, false), ()), [0 -1.0])
        ((((1, 1, 0.5), (0, 2, 0.0)), (1, 3, 0.5), (false, false), ()), (((0, 2, 0.0), (1, 1, 0.5)), (1, 3, 0.5), (false, false), ()), [0; 1/sqrt(2)])
        ((((1, 1, 0.5), (0, 2, 0.0)), (1, 3, 0.5), (false, false), ()), (((0, 2, 1.0), (1, 1, 0.5)), (1, 3, 0.5), (false, false), ()), [0; sqrt(3/2)])
    ]

    crOp = zeros(dataType, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
    # TODO: Check how can this be inherently defined. Maybe this forces the convention to be changed
    for (f1, f2, ftree_array) in crOp_ftree_nzdata

        v_in = f2[1][1]

        if v_in == (0, 0, 1.0)
            # Flip sign to the first col (0,0,1)_1 # Solves the addiitonal negative sign to the long cases of caac and acca by flipping signs to (0,0,1)_1 to An2 v_in and Cr2 v_out (or viceversa); in order to have anticommutivity in the (0,0,1)_1 virtual space, (+1, -1) with opposite signs
            ftree_array[:,1] *= -1
        elseif v_in == (0, 0, 0.0)
            # Flip sign to the second col (0,0,0)_2 # Solves the addiitonal negative sign to the long cases of caac and acca by flipping signs to (0,0,1)_1 to An2 v_in and Cr2 v_out (or viceversa); in order to have anticommutivity in the (0,0,1)_1 virtual space, (+1, -1) with opposite signs
            ftree_array[:,2] *= -1
        end


        crOp[ftree_type(f1...), ftree_type(f2...)][:,1,:,1] = ftree_array
    end


    # ANNIHILATION OPERATOR

    # New Convention + 2 multiplicities for (1,+-1,1//2)
    # No additional multiplicities con contraction logics
    anOp_ftree_nzdata=[
        ((((0, 0, 0.0), (0, 0, 0.0)), (0, 0, 0.0), (false, false), ()), (((1, -1, 0.5), (1, 1, 0.5)), (0, 0, 0.0), (false, false), ()), [sqrt(2) 0; 0 1])
        ((((1, -1, 0.5), (1, 1, 0.5)), (0, 0, 0.0), (false, false), ()), (((0, -2, 0.0), (0, 2, 0.0)), (0, 0, 0.0), (false, false), ()), [0; -1])
        ((((0, 0, 1.0), (0, 0, 0.0)), (0, 0, 1.0), (false, false), ()), (((1, -1, 0.5), (1, 1, 0.5)), (0, 0, 1.0), (false, false), ()), [0 1.0])
        ((((1, -1, 0.5), (1, 1, 0.5)), (0, 0, 1.0), (false, false), ()), (((0, -2, 1.0), (0, 2, 0.0)), (0, 0, 1.0), (false, false), ()), [0; -1.0])
        ((((1, 1, 0.5), (0, 0, 0.0)), (1, 1, 0.5), (false, false), ()), (((0, 0, 0.0), (1, 1, 0.5)), (1, 1, 0.5), (false, false), ()), [1.0 0; 0 1/sqrt(2)])
        ((((1, 1, 0.5), (0, 0, 0.0)), (1, 1, 0.5), (false, false), ()), (((0, 0, 1.0), (1, 1, 0.5)), (1, 1, 0.5), (false, false), ()), [0; sqrt(3/2)])
        ((((0, 0, 0.0), (1, 1, 0.5)), (1, 1, 0.5), (false, false), ()), (((1, -1, 0.5), (0, 2, 0.0)), (1, 1, 0.5), (false, false), ()), [-1.0 0; 0 -1/sqrt(2)])
        ((((0, 0, 1.0), (1, 1, 0.5)), (1, 1, 0.5), (false, false), ()), (((1, -1, 0.5), (0, 2, 0.0)), (1, 1, 0.5), (false, false), ()), [0 sqrt(3/2)])
        ((((1, -1, 0.5), (0, 0, 0.0)), (1, -1, 0.5), (false, false), ()), (((0, -2, 0.0), (1, 1, 0.5)), (1, -1, 0.5), (false, false), ()), [0; -1/sqrt(2)])
        ((((1, -1, 0.5), (0, 0, 0.0)), (1, -1, 0.5), (false, false), ()), (((0, -2, 1.0), (1, 1, 0.5)), (1, -1, 0.5), (false, false), ()), [0; sqrt(3/2)])
        ((((0, 2, 0.0), (0, 0, 0.0)), (0, 2, 0.0), (false, false), ()), (((1, 1, 0.5), (1, 1, 0.5)), (0, 2, 0.0), (false, false), ()), [0 -1])
        ((((1, 1, 0.5), (1, 1, 0.5)), (0, 2, 0.0), (false, false), ()), (((0, 0, 0.0), (0, 2, 0.0)), (0, 2, 0.0), (false, false), ()), [sqrt(2) 0; 0 1])
        ((((0, 2, 1.0), (0, 0, 0.0)), (0, 2, 1.0), (false, false), ()), (((1, 1, 0.5), (1, 1, 0.5)), (0, 2, 1.0), (false, false), ()), [0 1.0])
        ((((1, 1, 0.5), (1, 1, 0.5)), (0, 2, 1.0), (false, false), ()), (((0, 0, 1.0), (0, 2, 0.0)), (0, 2, 1.0), (false, false), ()), [0; -1.0])
        ((((0, 2, 0.0), (1, 1, 0.5)), (1, 3, 0.5), (false, false), ()), (((1, 1, 0.5), (0, 2, 0.0)), (1, 3, 0.5), (false, false), ()), [0 1/sqrt(2)])
        ((((0, 2, 1.0), (1, 1, 0.5)), (1, 3, 0.5), (false, false), ()), (((1, 1, 0.5), (0, 2, 0.0)), (1, 3, 0.5), (false, false), ()), [0 sqrt(3/2)])
    ]


    anOp = zeros(dataType, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
    for (f1, f2, ftree_array) in anOp_ftree_nzdata
        # Solves the single hopping sign problem in the upper triangular section of the h1e matrix. Now `parity_sign` is not required and properly handled.
        # Negate right VS for the annihilation operator, i.e. when the creation operator (of the same id -1 in this case-) is on the right.
        # Positive VS from the right, e.g.(1,1,0.5), do not change sign because there could not have been a creation to the right.

        v_in = f2[1][1]

        if v_in == (0, 0, 0.0)
            # Flip sign to the second column (0,0,0)_2
            ftree_array[:,2] = ftree_array[:,2] * -1
        elseif v_in == (0, 0, 1.0)
            # Flip sign to the first column (0,0,1)_1
            ftree_array[:,1] *= -1
        elseif v_in == (1, -1, 0.5)
            # Flip sign to the first column (1,+-1,0.5)_1
            ftree_array[:,1] *= -1
        elseif v_in == (0, -2, 0.0)
            # Flip sign to the first column (0,-2,0)_1
            ftree_array[:,1] *= -1
        elseif v_in == (0, -2, 1.0)
            # Flip sign to the first column (0,-2,0)_1
            ftree_array[:,1] *= -1
        end

        
        v_out = f1[1][1]
        if v_out == (0, 0, 1.0)
            # Flip sign to the first row (0,0,1)_1 # Solves the addiitonal negative sign to the long cases of caac and acca by flipping signs to (0,0,1)_1 to An2 v_in and Cr2 v_out (or viceversa); in order to have anticommutivity in the (0,0,1)_1 virtual space, (+1, -1) with opposite signs
            ftree_array[1,:] *= -1
        elseif v_out == (0, 0, 0.0)
            # Flip sign to the second row (0,0,0)_1 # Solves the addiitonal negative sign to the long cases of caac and acca by flipping signs to (0,0,1)_1 to An2 v_in and Cr2 v_out (or viceversa); in order to have anticommutivity in the (0,0,1)_1 virtual space, (+1, -1) with opposite signs
            ftree_array[2,:] *= -1
        end
        
        
        anOp[ftree_type(f1...), ftree_type(f2...)][:,1,:,1] = ftree_array
    end

    # IDENTITY OPERATOR
    idOp = localIdOp(phySpace, auxVecSpace; dataType=dataType)
    
    return LocalOps(crOp, anOp, idOp)
end

function localIdOp(phySpace, auxVecSpace; dataType::DataType=Float64)
    """ Construct local spatial orbital identity operator """

    dim_vSpace = TensorKit.dim(auxVecSpace)
    opData = zeros(dataType, dim_vSpace, 4, dim_vSpace, 4)
    for i in 1:dim_vSpace
        opData[i, :, i, :] = diagm(ones(dataType, 4))
    end
    
    idOp = TensorMap(opData, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
    return idOp
end

function ftree_inner_data(ftree)
    sectors_qn = [only(fieldnames(typeof(sector))) for sector in ftree.sectors]
    data = []
    for (qn, sector) in zip(sectors_qn, ftree.sectors)
        if qn == :isodd
            # push!(data, convert(Int64, sector.isodd))
            push!(data, sector.isodd)
        elseif qn == :charge
            # push!(data, convert(Int64, sector.charge))
            push!(data, sector.charge)
        elseif qn == :j
            # push!(data, convert(Float64, sector.j))
            push!(data, sector.j)
        else
            throw(ArgumentError("Unsupported sector type: $sector"))
        end
    end
    return tuple(data...)
end

"""
    FusionTreeDataType(f::FusionTree)
    Returns a tuple with the first element being the virtual sector and the second the fusion tree data for defining the fusion tree from its fusiontree type
    Tuple(
        VirtualSector,
        FusionTree data
    )
"""
ftree_data(f) = (ftree_inner_data(f.uncoupled[1]), (ftree_inner_data.(f.uncoupled), ftree_inner_data(f.coupled), (false, false), ()))

function FusionTreeDataType(QNType)
    return Tuple{
        Tuple{Tuple{QNType, QNType}, 
              QNType, 
              Tuple{Bool, Bool}, 
              Tuple{}},
        Tuple{Tuple{QNType, QNType}, 
              QNType, 
              Tuple{Bool, Bool}, 
              Tuple{}},
        Matrix{Float64}
    }
end

abstract type AbstractLocalOps{T} end

struct LocalOps{T} <: AbstractLocalOps{T}
    ops::Dict{String, TensorMap{T}}
    aliases::Dict{String, String}

    # For the symmetric constructor
    function LocalOps(c::TensorMap{T}, a::TensorMap{T}, I::TensorMap{T}) where T
        ops = Dict("c" => c, "a" => a, "I" => I)
        aliases = Dict{String, String}()
        return new{T}(ops, aliases)
    end

    # For the non-symmetric constructor
    function LocalOps(cu::TensorMap{T}, cd::TensorMap{T}, au::TensorMap{T}, ad::TensorMap{T}, I::TensorMap{T}) where T
        ops = Dict{String, TensorMap{T}}(
            "cu" => cu,
            "cd" => cd,
            "au" => au,
            "ad" => ad,
            "I" => I
        )
        aliases = Dict{String, String}(
            "c↑" => "cu",
            "c↓" => "cd", 
            "a↑" => "au",
            "a↓" => "ad"
        )
        return new{T}(ops, aliases)
    end
    
end

function Base.getindex(lo::LocalOps, key::String)
    if haskey(lo.ops, key)
        return lo.ops[key]

    elseif haskey(lo.aliases, key)
        return lo.ops[lo.aliases[key]]
    else
        available_keys = [collect(keys(lo.ops)); collect(keys(lo.aliases))] 
        throw(KeyError("LocalOps does not contain key '$key'. Available keys are: $(sort(available_keys))"))
    end
end

Base.keys(lo::LocalOps) = [collect(keys(lo.ops)); collect(keys(lo.aliases))]

struct LocalOps_DoubleV{T} <: AbstractLocalOps{T}
    base_ops::LocalOps{T}

    LocalOps_DoubleV(base_ops::LocalOps{T}) where T = new{T}(base_ops)
end

function Base.getindex(lo2V::LocalOps_DoubleV, key::String)
    # TODO: "c2" and "a2" are defined on-the-fly to save up memory, but it may actually be better to precompute them and save them in the base_ops dictionary.
    if key == "I"
        return lo2V.base_ops["I"]
    elseif key == "c1"
        return lo2V.base_ops["c"]
    elseif key == "c2"
        c = copy(lo2V.base_ops["c"])

        # Check the codomain's parity. If it's zero, then the domain's must be one (and vice versa).
        # We swap the (2-sized) dimension where parity is 1. This is in order to match the id=2 operator to the second (1,1,0.5) multiplicity.
        # Then, in order to add anticommutivity between the operators with id 1 and 2:
        #    - Add a negative sign to the second multiplicity of the (0,0,0) sector, which represents (1,-1) or (-1,1) in the virtual IDs quantum numbers with the same spin.
        #    - Add a negative sign to the (0,+-2,0) sector to distinguish between the (1,1) cases with opposite spins.
        #    - Add a negative sign to the (0,2,0) sector to distinguish between the (1,1) cases with opposite spins.
        for (f1, f2) in fusiontrees(c)
            ftree_array = c[f1,f2]
            if !iszero(ftree_array)
                v_out = ftree_data(f1)[1]
                if iszero(v_out[1])
                    # Swap columns (domain)
                    c[f1,f2][:,1,:,1] = ftree_array[:,1,[2,1],1]

                    if v_out == (0, -2, 1.0)
                        # Flip sign to the first row (0,-2,1)_1
                        c[f1,f2][1,1,:,1] *= -1
                    end
                else
                    # Swap rows (codomain)
                    c[f1,f2][:,1,:,1] = ftree_array[[2,1],1,:,1]
                    
                    v_in = ftree_data(f2)[1]

                    if v_in == (0, 2, 1.0)
                        # Flip sign to the first column (0,2,1)_1
                        c[f1,f2][:,1,1,1] *= -1
                    end
                end
            end
        end

        return c
    elseif key == "a1"
        return lo2V.base_ops["a"]
    elseif key == "a2"
        a = copy(lo2V.base_ops["a"])

        for (f1, f2) in fusiontrees(a)
            ftree_array = a[f1,f2]
            if !iszero(ftree_array)
                v_out = ftree_data(f1)[1]
                if iszero(v_out[1])
                    # Swap columns (domain)
                    a[f1,f2][:,1,:,1] = ftree_array[:,1,[2,1],1]

                    if v_out == (0, 2, 1.0)
                        # Flip sign to the first row (0,2,1)_1
                        a[f1,f2][1,1,:,1] *= -1
                    end
                else
                    # Swap rows (codomain)
                    a[f1,f2][:,1,:,1] = ftree_array[[2,1],1,:,1]
                    
                    v_in = ftree_data(f2)[1]

                    if v_in == (0, -2, 1.0)
                        # Flip sign to the first column (0,-2,1)_1
                        a[f1,f2][:,1,1,1] *= -1
                    end
                end
            end
        end
        return a
    else
        throw(ArgumentError("Unsupported operator key: $key. Supported keys are: 'I', 'c1', 'c2', 'a1', 'a2'."))
    end
end

Base.keys(lo2V::LocalOps_DoubleV) = ["I", "c1", "c2", "a1", "a2"]


struct Op2Data{T}
    all_ops_to_data_dict::T
    ops::AbstractLocalOps{Float64}
    symm::String
    
    function Op2Data(phySpace, QNType, spin_symm::Bool)
        # Define the OpDataDict type
        
        OpDataDict = Dict{String, Dict{Tuple{QNType, QNType}, Vector{FusionTreeDataType(QNType)}}}

        # Get the local operators based on the sector type of the physical space
        sector_type = TensorKit.type_repr(sectortype(phySpace))
        if sector_type in ("(FermionParity ⊠ U1Irrep ⊠ U1Irrep)", "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])")
            ops = GetCrAnLocalOpsU1U1(phySpace, spin_symm=spin_symm)
            symm = "u1u1"
        elseif sector_type in ("(FermionParity ⊠ U1Irrep ⊠ SU2Irrep)", "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])")
            # spin_symm is irrelevant, as SU2 is inherently spin symmetric. Add a warning?
            ops = GetCrAnLocalOpsU1SU2(phySpace)
            ops_doubleV = LocalOps_DoubleV(ops)
            symm = "u1su2"
        else
            throw(ArgumentError("Unsupported sector type: $sector_type"))
        end
        return new{OpDataDict}(OpDataDict(), ops_doubleV, symm)
    end
end

function Base.getindex(op2data::Op2Data, input::Tuple{String, QN, QN}) where QN
    op_str, left_qn_in, right_qn_in = input

    if haskey(op2data.all_ops_to_data_dict, op_str)
        if haskey(op2data.all_ops_to_data_dict[op_str], (left_qn_in, right_qn_in))
            return op2data.all_ops_to_data_dict[op_str][(left_qn_in, right_qn_in)]
        else
            @warn "Operator $op_str with left QN $left_qn_in and right QN $right_qn_in not found in the dictionary."
            return FusionTreeDataType(QN)[]
        end
    else
        op2data.all_ops_to_data_dict[op_str] = Dict{Tuple{QN, QN}, Vector{FusionTreeDataType(QN)}}()

        # Construct the operator's TensorMap
        # Note that the order of the operations is flipped when constructing the composite operator
        # For example, if the operator is "ac", first the annihilation is applied and then the creation
        # So that the composite operator will have the creation operator's domain and the annihilation operator's codomain
        op_TM = construct_op_TensorMap(op2data, op_str)

        # Construct the list of the operator's non-zero fusion trees
        for (f1, f2) in fusiontrees(op_TM)
            val = Matrix(op_TM[f1,f2][:,1,:,1]) # Since there are no multiplicities in the virtual space, we get the fusion tree values as a matrix with dimension (#left virtual multiplicities, #right virtual multiplicities)
            if !all(val .== 0)
                left_qn_, ftree_left  = ftree_data(f1)
                right_qn_, ftree_right = ftree_data(f2)

                if !haskey(op2data.all_ops_to_data_dict[op_str], (left_qn_, right_qn_))
                    op2data.all_ops_to_data_dict[op_str][(left_qn_, right_qn_)] = FusionTreeDataType(QN)[]
                end
                # println("Adding operator $op_str with left QN $left_qn_ and right QN $right_qn_ to the dictionary.")

                push!(op2data.all_ops_to_data_dict[op_str][(left_qn_, right_qn_)], (ftree_left, ftree_right, val))
            end
        end

        return op2data.all_ops_to_data_dict[op_str][(left_qn_in, right_qn_in)]
    end
end

function construct_op_TensorMap(op2data::Op2Data, op_str::String)
    """
    Construct a TensorMap for a given operator string using the Op2Data structure.
    
    Parameters:
    - op2data: Op2Data instance containing operator data
    - op_str: String representing the operator
    
    Returns:
    - TensorMap representing the operator
    """
    if op_str == "I"
        # Identity operator
        return op2data.ops["I"]
    end
    if op2data.symm == "u1su2"
        if any(c -> c in "12", op_str)
            op_TM = reduce(*, [op2data.ops[join(collect(op_str)[i:i+1])] for i in reverse(1:2:length(op_str))])
            # op_TM = construct_op(op2data.ops, op_str)
        elseif any(c -> c in "↑↓", op_str)
            # Spin symmetric operators are constructed without taking the spin labels into account
            # For example, "a↑c↑" or "a↓c↓" will be constructed as "ac"
            op_TM = reduce(*, [op2data.ops[string(collect(op_str)[i])] for i in reverse(1:2:length(op_str))])
        else
            # String is already in the form of "ac", "ca", etc. Without spin labels
            op_TM = reduce(*, [op2data.ops[string(char)] for char in reverse(op_str)])
        end
    elseif op2data.symm == "u1u1"
        op_TM = reduce(*, [op2data.ops[join(collect(op_str)[i:i+1])] for i in reverse(1:2:length(op_str))])
    else
        throw(ArgumentError("Unsupported symmetry: $(op2data.symm)"))
    end
    
    return op_TM
end

function construct_op(ops, op_str::String)
    if op_str == "I"
        # Identity operator
        return op2data.ops["I"]
    end
    if length(op_str) == 2
        # Single operator
        return ops[op_str]
    end

    op_TM = ops[join(collect(op_str)[1:2])]

    # Construct the operator to take care of the crossing between virtual and physical spaces
    auxVecSpace = TensorKit.space(op_TM, 1)
    phySpace = TensorKit.space(op_TM, 2)
    # crossing_op = isometry(phySpace ⊗ auxVecSpace, fuse(auxVecSpace, phySpace)) * isometry(fuse(auxVecSpace, phySpace), auxVecSpace ⊗ phySpace)
    crossing_op = isometry(phySpace ⊗ auxVecSpace, auxVecSpace ⊗ phySpace)

    # for i in reverse(1:2:(length(op_str)-2))
    for i in 3:2:length(op_str)
        single_op_str = join(collect(op_str)[i:i+1])
        op_TM = TensorKit.permute(ops[single_op_str], (1,2), (4,3)) * crossing_op * op_TM
    end
    return op_TM
end

# Remove this function when the MPO construction is stable.
function remove_approximate_match!(array, target_tuple, rtol=1e-10)
    # Unpack the target tuple elements
    target_ftree_left, target_ftree_right, target_mult_left, target_mult_right, target_value = target_tuple
    
    # Filter the array in-place
    filter!(x -> begin
        # Check if first 4 elements match exactly
        if x[1] == target_ftree_left && 
           x[2] == target_ftree_right && 
           x[3] == target_mult_left && 
           x[4] == target_mult_right
            # For the last element, check with isapprox
            return !isapprox(x[5], target_value, rtol=rtol)
        end
        return true
    end, array)
end


function fill_mpo_site_SU2_doubleV!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type; spin_symm=true, verbose=true)

    for (row, col, ops) in zip(findnz(symb_mpo_site)...)

        for (left_qn, left_op_mult, left_MPOsite_mult) in vs_map_left[row]
            for (right_qn, right_op_mult, right_MPOsite_mult) in vs_map_right[col]
        
                for op in ops

                    # TODO: Remove when the MPO construction is stable. spin_symm should be always true for SU2.
                    if spin_symm
                        # TODO: Check if this is needed or if there is a better way to handle the operator string for U1U1
                        op_str = op.operator == "I" ? "I" : join([collect(op.operator)[i] for i in 1:2:length(op.operator)]) # Remove the spin labels from the operator string
                    else
                        op_str = op.operator
                    end

                    op_data = op2data[(op_str, left_qn, right_qn)]

                    for (ftree_left, ftree_right, val) in op_data
                        f1 = ftree_type(ftree_left...)
                        f2 = ftree_type(ftree_right...)

                        verbose && println("Filling MPO site for operator: $op, op_str $op_str, left QN: $left_qn, left_op_mult: $left_op_mult, right QN: $right_qn, right_op_mult: $right_op_mult left_mult: $left_MPOsite_mult, right_mult: $right_MPOsite_mult, ftree_left: $ftree_left, ftree_right: $ftree_right, val: $val, val[left_op_mult, right_op_mult]: $(val[left_op_mult, right_op_mult]),  coef: $(op.coefficient)")
                        mpo_site[f1,f2][left_MPOsite_mult, 1, right_MPOsite_mult, 1] += val[left_op_mult, right_op_mult] * op.coefficient # We should be filling the mpo_site[f1,f2][mult_left,:,mult_right,:] combination twice due to symmetry
                        # mpo_site[f1,f2][mult_left,:,mult_right,:] .= val * coef # This is the original line, but we need to check for symmetry, as we should be filling the mpo_site[f1,f2][mult_left,:,mult_right,:] combination twice
                        
                        # @assert length(mpo_site[f1,f2][left_mult,:,right_mult,:]) == 1 # Remove when the MPO construction is stable. This is a sanity check to ensure that the MPO site has the expected shape.
                    end
                end
            end

        end
    end

end

"""
    convert_to_tensorkit_mpo(symbolic_mpo, mpoVs, phySpace::GradedSpace, dataType::DataType=Float64)

Convert a symbolic MPO representation to TensorKit's SparseMPO format.
This transforms the symbolic MPO created by the graph-based optimization into
actual tensor objects that can be used for calculations.

Parameters:
- symbolic_mpo: Symbolic MPO from construct_symbolic_mpo
- mpoVs: Virtual spaces for the MPO
- phySpace: Physical space (GradedSpace) for TensorKit tensors
- dataType: Data type for tensor elements

Returns:
- TensorKit MPO representation of the Hamiltonian
"""
function fill_mpo_site_SU2!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type, QNType)

    symbolic_operators_iter = get_symbolic_operators_iter(symb_mpo_site, vs_map_left, vs_map_right)

    # Make sure the numeric MPO is filled properly: every operator has its spin symmetric counterpart.
    # TODO: Remove this check when the MPO construction is stable.
    symm_check = []
    for (left_qn, left_mult, m_Vleft, right_qn, right_mult, m_Vright, coef, op_str) in symbolic_operators_iter

        # coef = op.coefficient == 1 ? op.coefficient : op.coefficient / 2 # If the coefficient is not 1, we need to divide it by 2 to account for the symmetry (as we fill the MPO site twice)
        # symm_op_string = op.operator == "I" ? "I" : join([collect(op.operator)[i] for i in 1:2:length(op.operator)]) # Remove the spin labels from the operator string
        
        
        op_data = FusionTreeDataType(QNType)[]  # Initialize op_data before try block
        try
            op_data = op2data[(op_str, left_qn, right_qn)]
        catch e
            println("Failed to find data for operator: '$op_str'")
            println("Left QN: $left_qn, mult=$left_mult, left magnetic projection=$m_Vleft (currently unused)")
            println("Right QN: $right_qn, mult=$right_mult, right magnetic projection=$m_Vright (currently unused)")
            
            rethrow(e)
        end

        if isempty(op_data)
            @warn "No data found for operator $op_str with left QN $(left_qn) and right QN $(right_qn)."
            continue
        end
        for (ftree_left, ftree_right, val) in op_data
            f1 = ftree_type(ftree_left...)
            f2 = ftree_type(ftree_right...)

            mpo_site[f1,f2][left_mult,:,right_mult,:] .+= val * coef # We should be filling the mpo_site[f1,f2][mult_left,:,mult_right,:] combination twice due to symmetry
            # mpo_site[f1,f2][mult_left,:,mult_right,:] .= val * coef # This is the original line, but we need to check for symmetry, as we should be filling the mpo_site[f1,f2][mult_left,:,mult_right,:] combination twice
            
            @assert length(mpo_site[f1,f2][left_mult,:,right_mult,:]) == 1 # Remove when the MPO construction is stable. This is a sanity check to ensure that the MPO site has the expected shape.
            # # Remove this section when the MPO construction is stable? Still would need to take care of the double value setting due to symmetry.
            # # Remove FROM HERE
            # if symm_op_string == "I"
            #     # No need to check for symmetry while filling for the identity operator
            #     mpo_site[f1,f2][mult_left,:,mult_right,:] .= val * coef
            # else
            #     if only(mpo_site[f1,f2][mult_left,:,mult_right,:]) == 0
            #         mpo_site[f1,f2][mult_left,:,mult_right,:] .= val * coef
            #         push!(symm_check, (ftree_left, ftree_right, mult_left, mult_right, val*coef))
            #     else
            #         # Make sure the ftrees+value pair is already set (and once only)
            #         # @assert !symm_check[(ftree_left, ftree_right, mult_left, mult_right, val*coef)]

            #         remove_approximate_match!(symm_check,(ftree_left, ftree_right, mult_left, mult_right, val*coef))
            #     end
            # end
            # # Remove this section when the MPO construction is stable? UNTIL HERE
        end
    end

    # Check that all the ftrees+value pairs were set symmetrically
    @assert isempty(symm_check) "Not all ftrees+value pairs were set in the MPO site. This suggests an imbalance in the Hamiltonian terms."
    
end

function fill_mpo_site_U1U1!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type, spin_symm)

    for (row, col, ops) in zip(findnz(symb_mpo_site)...)
        (left_qn, left_mult) = vs_map_left[row]
        (right_qn, right_mult) = vs_map_right[col]
        
        for op in ops

            if spin_symm
                # TODO: Check if this is needed or if there is a better way to handle the operator string for U1U1
                op_str = op.operator == "I" ? "I" : join([collect(op.operator)[i] for i in 1:2:length(op.operator)]) # Remove the spin labels from the operator string
            else
                op_str = op.operator
            end

            op_data = op2data[(op_str, left_qn, right_qn)]

            if isempty(op_data)
                @warn "No data found for operator $op.operator with left QN $(left_qn) and right QN $(right_qn)."
                continue
            end
            for (ftree_left, ftree_right, val) in op_data
                f1 = ftree_type(ftree_left...)
                f2 = ftree_type(ftree_right...)

                # println("Filling MPO site for operator: $op, op_str $op_str, left QN: $left_qn, right QN: $right_qn, left_mult: $left_mult, right_mult: $right_mult, ftree_left: $ftree_left, ftree_right: $ftree_right, value: $val, coef: $(op.coefficient)")
                mpo_site[f1,f2][left_mult,:,right_mult,:] .+= val * op.coefficient # We should be filling the mpo_site[f1,f2][mult_left,:,mult_right,:] combination twice due to symmetry
                # mpo_site[f1,f2][mult_left,:,mult_right,:] .= val * coef # This is the original line, but we need to check for symmetry, as we should be filling the mpo_site[f1,f2][mult_left,:,mult_right,:] combination twice
                
                @assert length(mpo_site[f1,f2][left_mult,:,right_mult,:]) == 1 # Remove when the MPO construction is stable. This is a sanity check to ensure that the MPO site has the expected shape.
            end
        end
    end

end

function convert_to_tensorkit_mpo(symbolic_mpo, virt_spaces, phySpace::GradedSpace; spin_symm::Bool=false, double_vSpace=false, dataType::DataType=Float64, verbose=true)
    # TODO: Remove spin_symm parameter when the MPO construction is stable. It should always be false for U1U1 and true for U1SU2.

    mpo_sites = Vector{TensorMap}(undef, length(symbolic_mpo)) # Allocate TensorMap with the already known properties (type, shape, etc.)
    phySpace_dim = TensorKit.dim(phySpace)
    @assert phySpace_dim == 4
    symm = TensorKit.type_repr(sectortype(phySpace))
    qn_parser = QNParser(symm, double_vSpace=double_vSpace)
    if double_vSpace
        QNType = typeof(qn_parser(0,0)[1]) # For double virtual spaces, the QNType is the quantum number type of any of the virtual spaces, as they get fused together
    else
        QNType = typeof(qn_parser(0,0))
    end
    qn_vs_maps, qn_mult_counts = get_QN_mapping_and_vs_multiplicity(symbolic_mpo, virt_spaces, qn_parser; verbose=1, symm=symm, double_vSpace=double_vSpace)

    op2data = Op2Data(phySpace, QNType, spin_symm)
    ftree_type = FusionTree{sectortype(phySpace)}

    for (isite, symb_mpo_site) in enumerate(symbolic_mpo)
        qn_mult_counts_left = qn_mult_counts[isite]
        qn_mult_counts_right = qn_mult_counts[isite+1]
        mpo_site = construct_empty_mpo_site(phySpace, qn_mult_counts_left, qn_mult_counts_right; dataType=dataType)
        
        vs_map_left = qn_vs_maps[isite]
        vs_map_right = qn_vs_maps[isite+1]
        
        if symm in ("(FermionParity ⊠ U1Irrep ⊠ SU2Irrep)", "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])")
            if double_vSpace
                fill_mpo_site_SU2_doubleV!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type; spin_symm=spin_symm, verbose=verbose)
            else
                fill_mpo_site_SU2!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type, QNType)
            end
        elseif symm in ("(FermionParity ⊠ U1Irrep ⊠ U1Irrep)", "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])")
            fill_mpo_site_U1U1!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type, spin_symm)
        else
            throw(ArgumentError("Unsupported symmetry type: $symm"))
        end
        
        mpo_sites[isite] = mpo_site
    end

    # Create the final MPO object
    mpo = SparseMPO([mpo_sites...])
    
    return mpo
end

"""
    genHamMPO_bpt(chem_data, ord; phySpace=nothing, ops_tol=1e-14, maxdim=2^30, dataType::DataType=Float64)

Generate a Hamiltonian MPO using the bipartite graph optimization approach.
This is the main function that brings together all components.

Parameters:
- chem_data: Chemical data containing Hamiltonian coefficients
- ord: Orbital ordering
- phySpace: Physical space for TensorKit tensors (optional)
- ops_tol: Tolerance for discarding small terms
- maxdim: Maximum bond dimension
- dataType: Data type for tensor elements

Returns:
- An MPO representation of the Hamiltonian, either symbolic or as a TensorKit MPO
"""
function genHamMPO_bpt(chem_data, ord, phySpace; ops_tol=1e-14, maxdim=2^30, dataType::DataType=Float64, algo="Hungarian", spin_symm::Bool=false, verbose=true)
    # Generate the OpSum object from the Hamiltonian coefficients:
    terms = gen_OpSum(chem_data, ord; tol=ops_tol, spin_symm=spin_symm)

    # Convert to operator table
    table, primary_ops, factors = _terms_to_table(chem_data.N_spt, terms)
    
    # Build symbolic MPO with graph-based optimization
    symbolic_mpo, virt_spaces = construct_symbolic_mpo(table, primary_ops, factors; algo=algo, verbose=verbose)
    
    return convert_to_tensorkit_mpo(symbolic_mpo, virt_spaces, phySpace; spin_symm=false, double_vSpace=false, dataType=dataType, verbose=verbose)
end
