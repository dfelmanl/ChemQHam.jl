
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

function Base.:/(term::OpTerm, scalar::Number)
    # Divide the coefficient of the term by a scalar
    return OpTerm(term.coefficient / scalar, term.operators, term.sites)
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

# ChemOpSum structure to collect and manage operator terms
struct ChemOpSum
    terms::Vector{OpTerm}
    
    ChemOpSum() = new(OpTerm[])
    ChemOpSum(terms::Vector{OpTerm}) = new(terms)
    ChemOpSum(term::OpTerm) = new([term])
end

# Add a term to the ChemOpSum
function Base.:+(opsum::ChemOpSum, args)
    if length(args) % 2 == 1 && !isa(args[1], Number)
        throw(ArgumentError("Expected coefficient followed by operator/site pairs"))
    end
    
    coef = args[1]
    ops_sites = args[2:end]
    
    term = OpTerm(coef, ops_sites...)
    push!(opsum.terms, term)
    
    return opsum
end

function Base.:+(opsum::ChemOpSum, term::OpTerm)
    # Add a single OpTerm to the ChemOpSum
    push!(opsum.terms, term)
    return opsum
end

function Base.:+(opsum::ChemOpSum, terms::Vector{OpTerm})
    # Add multiple OpTerms to the ChemOpSum
    for term in terms
        push!(opsum.terms, term)
    end
    return opsum
end

function Base.:+(opsum1::ChemOpSum, opsum2::ChemOpSum)
    # Combine two OpSums
    new_terms = vcat(opsum1.terms, opsum2.terms)
    return ChemOpSum(new_terms)
end

function Base.:*(opsum::ChemOpSum, scalar::Number)
    # Multiply all terms in the ChemOpSum by a scalar
    new_terms = [term * scalar for term in opsum.terms]
    return ChemOpSum(new_terms)
end

Base.:*(scalar::Number, opsum::ChemOpSum) = opsum * scalar

# Iterator interface for ChemOpSum
Base.getindex(ops::ChemOpSum, i::Int) = ops.terms[i]
Base.getindex(ops::ChemOpSum, r::AbstractRange) = ChemOpSum(ops.terms[r])
Base.getindex(ops::ChemOpSum, indices) = ChemOpSum(ops.terms[collect(indices)])
Base.iterate(ops::ChemOpSum) = isempty(ops.terms) ? nothing : (ops.terms[1], 1)
Base.keys(ops::ChemOpSum) = 1:length(ops.terms)
Base.iterate(ops::ChemOpSum, state) = state >= length(ops.terms) ? nothing : (ops.terms[state+1], state+1)
Base.length(ops::ChemOpSum) = length(ops.terms)
Base.eltype(::Type{ChemOpSum}) = OpTerm

function Base.show(io::IO, ops::ChemOpSum)
    if isempty(ops.terms)
        print(io, "ChemOpSum: (empty)")
    else
        print(io, "ChemOpSum with $(length(ops.terms)) terms:\n")
        for (i, term) in enumerate(ops.terms)
            print(io, "  $i. ")
            show(io, term)
            if i < length(ops.terms)
                print(io, "\n")
            end
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", ops::ChemOpSum)
    show(io, ops)
end


gen_ChemOpSum(molecule::Molecule; kwargs...) = gen_ChemOpSum(xyz_string(Molecule(molecule)); kwargs...)
function gen_ChemOpSum(mol_str::String; kwargs...)
    h1e, h2e, nuc_e, hf_orb_occ_basis, hf_elec_occ, hf_energy = molecular_hf_data(mol_str)
    return gen_ChemOpSum(h1e, h2e, nuc_e; kwargs...)
end
gen_ChemOpSum(h1e, h2e, nuc_e; kwargs...) = gen_ChemOpSum(h1e, h2e; nuc_e=nuc_e, kwargs...)
gen_ChemOpSum(h1e, h2e, nuc_e, ord::Vector{Int}; kwargs...) = gen_ChemOpSum(h1e, h2e; nuc_e=nuc_e, ord=ord, kwargs...)
gen_ChemOpSum(h1e, h2e, nuc_e, n_sites::Int; kwargs...) = gen_ChemOpSum(h1e, h2e; nuc_e=nuc_e, n_sites=n_sites, kwargs...)

function gen_ChemOpSum(h1e::AbstractArray{Float64}, h2e::AbstractArray{Float64}; nuc_e=0.0, n_sites=0, ord=nothing, tol=1e-14, spin_symm::Bool=true, add_nuc::Bool=true)
    add_nuc || (nuc_e = 0.0)  # If add_nuc is false, set nuclear energy to 0
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

    # Generate the ChemOpSum object from the Hamiltonian coefficients
    if spin_symm
        # return _gen_OpSum_SpinSymm(h1e, h2e, nuc_e, ord; tol=tol) # Unprocessed h1e and h2e
        return _gen_OpSum_SpinSymm(h1e, h2e, nuc_e, ord; tol=tol) # Always assume that h1e and h2e are already processed when they are passed to gen_ChemOpSum?
    else
        # return _gen_OpSum_noSpinSymm(h1e, h2e, nuc_e, ord; tol=tol)
        return _gen_OpSum_noSpinSymm(h1e, h2e, nuc_e, ord; tol=tol)
    end
end

# Generate the ChemOpSum object from the Hamiltonian coefficients:
function _gen_OpSum_SpinSymm(h1e, h2e, nuc_e, ord; tol=1e-14)
    N_spt = length(ord)

    os = ChemOpSum()

    # Nuclear energy term
    if nuc_e != 0.0
        # Add the nuclear energy term as an identity operator on all sites
        os += nuc_e, ["I"], [0]
    end

    # One-interaction terms
    for p = 1:N_spt, q = 1:N_spt
        cf = h1e[ord[p], ord[q]]

        if abs(cf) >= tol
            if p + q - 1 <= N_spt # TODO: Check if it improves performance because of the bipartite grouping 
                os += cf, "a1", p, "c1", q  # Spin-up operators
            else
                os += cf, "a2", p, "c2", q  # Spin-down operators
            end
        end
    end

    # Two-interactions terms
    for p = 1:N_spt, q = p:N_spt, r = 1:N_spt
        if p == q
            s_start = r
        else
            s_start = 1
        end
        for s = s_start:N_spt

            cf = h2e[ord[p], ord[q], ord[r], ord[s]]

            if abs(cf) >= tol
                if !(p == q && r == s)
                    cf *= 2
                end
                os += cf, "a1", p, "a2", q, "c2", r, "c1", s # check if exists in the dict and flip
            end
        end
    end
    return os
end


function _gen_OpSum_noSpinSymm(h1e, h2e, nuc_e, ord; tol=1e-14)
    throw(ErrorException("ChemOpSum generation without spin symmetry is not supported yet. Choose spin_symm=true instead."))
end