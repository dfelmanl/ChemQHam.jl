
# Convert ChemOpSum terms to an operator table (similar to Renormalizer's _terms_to_table)
function terms_to_table(terms::ChemOpSum; n_sites::Int=0)
    n_sites = n_sites > 0 ? n_sites : maximum(maximum.(getfield.(terms, :sites)))  # Determine number of sites if not provided
    # TODO: Allow to get only `terms` and extract the number of sites from the terms themselves (maximum site location number).
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
    
    # Process each term in the ChemOpSum
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
