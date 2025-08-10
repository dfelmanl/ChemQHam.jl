
# Convert ChemOpSum terms to an operator table
function terms_to_table(
    terms::ChemOpSum,
    symm_ctx::AbstractSymmetryContext;
    n_sites::Int = 0,
)

    # Determine number of sites if not provided
    n_sites = n_sites > 0 ? n_sites : maximum(maximum.(getfield.(terms, :sites)))

    # Initialize the table where each row corresponds to a term
    table = Vector{Vector{Int}}()

    # Factor list to store each term's coefficient
    factor_list = Vector{Float64}()

    # Get the symmetry context's local operators index mapping
    localOps_idx_map = symm_ctx.local_ops_idx_map

    dummy_table_entry = fill(localOps_idx_map["I"], n_sites)

    for term in terms

        coef = term.coefficient
        site_ops = group_operators_by_site(term)

        table_entry = copy(dummy_table_entry)

        # Process each site in the term
        for (op_str, site) in site_ops

            # Update the table entry for this site
            op_idx = localOps_idx_map[op_str]
            table_entry[site] = op_idx

        end

        push!(table, table_entry)
        push!(factor_list, coef)
    end

    # Deduplicate table entries and combine factors
    table, factors = _deduplicate_table(table, factor_list)

    # Convert table to matrix
    table = reduce(vcat, [row' for row in table])

    return table, factors
end


function group_operators_by_site(term::OpTerm)
    # Create a dictionary to collect operators at each site
    site_ops_dict = Dict{Int,Vector{String}}()

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


function _deduplicate_table(table::Vector{Vector{Int}}, factor::Vector{Float64})
    # Create a dictionary to store unique table entries
    unique_entries = Dict{Vector{Int},Float64}()

    # Combine entries with the same operator configuration
    for i = 1:length(table)
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
