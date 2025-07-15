
"""
    construct_symbolic_mpo
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
function construct_symbolic_mpo(table, factors, localOps_idx_map, vsQN_idx_map, op2data; algo="Hungarian", verbose=false)

    n_sites = size(table, 2)

    idx_vsQN_map = Dict(v => k for (k, v) in vsQN_idx_map)
    idx_localOps_map = Dict(v => k for (k, v) in localOps_idx_map)
    
    # Add ones at the beginning and end of each row. They will be the index of the trivial auxiliary virtual bonds at the start and end of the MPO
    ones_col = ones(Int, size(table, 1))
    table = hcat(ones_col, table, ones_col)
    
    virtSpace_left = [vsQN_idx_map[((false, 0, 0), 1)]] # Start with the trivial virtual space. Later on, we can impose that the symmetry related idx_vsQN_map with have <trivial_space> as the first element; and just write virtSpace_in = [1]
    site_entries_list = Vector{Vector{Tuple{Int, Int, Int, Float64}}}(undef, n_sites)

    # Store the list of virtual spaces at each site
    virtSpaces_list = [virtSpace_left]

    verbose && println("Using $(algo) algorithm for bipartite matching optimization")
    
    # This is the main loop in Renormalizer's construct_symbolic_mpo
    for isite in 1:n_sites
        verbose && println("Processing site $(isite) of $(n_sites)")
        # Split table into row and column parts - always take first two columns
        # for rows and the rest for columns, as in Renormalizer
        table_row = table[:, 1:2]
        table_col = table[:, 3:end]
        
        # Call the one_site function to process this site
        site_entries, virtSpace_right, table, factors = _construct_symbolic_mpo_one_site(
            table_row, table_col, virtSpace_left, factors, idx_localOps_map, idx_vsQN_map, op2data; algo=algo
        )
        
        # Update for next iteration
        virtSpace_left = virtSpace_right
        
        # Store the virtual spaces for this site
        push!(virtSpaces_list, virtSpace_right)

        site_entries_list[isite] = site_entries
    end
    
    # At the end, we should have a single term with factor 1 (or close to it due to floating point)
    # Comment out assert for now during development
    @assert size(table, 1) == length(factors) == 1 && isapprox(factors[1], 1.0, atol=1e-10)
    @assert length(virtSpaces_list) == n_sites + 1

    # Construct symbolic MPO
    symbolic_mpo = [] # Preallocate if possible
    for isite in 1:n_sites
        symbolic_site = compose_symbolic_site_sparse(site_entries_list[isite], virtSpaces_list[isite], virtSpaces_list[isite+1], idx_localOps_map)
        push!(symbolic_mpo, symbolic_site)
    end


    # Calculate the virtual spaces' quantum numbers
    QNType = Tuple{Bool, Int, Rational{Int}} # Fix later to be defined by the symm obj

    mpoVs = Vector{Vector{Tuple{QNType, Int}}}(undef, length(virtSpaces_list))
    for (i, virtSpace_out) in enumerate(virtSpaces_list)
        mpoVs[i] = [idx_vsQN_map[vs_grp] for vs_grp in virtSpace_out]
    end

    # @assert all(length(unique(vs))==1 for vs_grp in mpoVs for vs in vs_grp)
    verbose && println("symbolic MPO's bond dimensions: $([length(vs) for vs in mpoVs])")
    
    return symbolic_mpo, mpoVs, virtSpaces_list
end

function construct_symbolic_mpo(op_terms::ChemOpSum, op2data::Op2Data; kwargs...)
    table, factors, localOps_idx_map, vsQN_idx_map = terms_to_table(op_terms, op2data)
    return construct_symbolic_mpo(table, factors, localOps_idx_map, vsQN_idx_map, op2data; kwargs...)
end

construct_symbolic_mpo(molecule::Molecule; kwargs...) = construct_symbolic_mpo(xyz_string(Molecule(molecule)); kwargs...)
construct_symbolic_mpo(mol_str::String; kwargs...) = construct_symbolic_mpo(molecular_interaction_coefficients(molecule)...; kwargs...)
construct_symbolic_mpo(h1e::AbstractArray{Float64}, h2e::AbstractArray{Float64}, nuc_e::Float64; kwargs...) = construct_symbolic_mpo(h1e, h2e; nuc_e=nuc_e, kwargs...)

function construct_symbolic_mpo(h1e::AbstractArray{Float64}, h2e::AbstractArray{Float64}; nuc_e::Float64=0.0, symm::String="U1SU2", ord=nothing, ops_tol=1e-14, maxdim=2^30, algo="Hungarian", spin_symm::Bool=true, verbose=false)
    
    terms = gen_ChemOpSum(h1e, h2e; nuc_e=nuc_e, n_sites=0, ord=ord, tol=ops_tol, spin_symm=spin_symm)
    op2data = Op2Data(symm)

    table, factors, localOps_idx_map, vsQN_idx_map = terms_to_table(terms, op2data)
    symbolic_mpo, virt_spaces = construct_symbolic_mpo(table, factors, localOps_idx_map, vsQN_idx_map, op2data; algo=algo, verbose=verbose)
    
    return symbolic_mpo, virt_spaces
end


function _construct_symbolic_mpo_one_site(table_row, table_col, virtSpace_left, factors, idx_localOps_map, idx_vsQN_map, op2data; algo="Hungarian")
    # Find unique rows and their inverse mapping
    term_row, row_unique_inverseMap = find_unique_with_inverseMap(table_row)
    
    # Make sure the dimensions match
    @assert size(table_row, 2) == 2
    
    # Find unique columns and their inverse mapping
    term_col, col_unique_inverseMap = find_unique_with_inverseMap(table_col)
    
    # Create a sparse matrix directly where non-zero values are indices into the factor array
    non_red = SparseArrays.sparse(row_unique_inverseMap, col_unique_inverseMap, 1:length(factors))
    
    site_entries, virtSpaces_out, table, new_factor = _decompose_graph(term_row, term_col, non_red, virtSpace_left, factors, idx_localOps_map, idx_vsQN_map, op2data, algo)
    
    return site_entries, virtSpaces_out, table, new_factor
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
function _decompose_graph(term_row, term_col, non_red, virtSpace_left_arr, factors, idx_localOps_map, idx_vsQN_map, op2data, algo)
    
    # Get dimensions directly from the sparse matrix
    n_rows, n_cols = size(non_red)

    vsQN_idx_map = get_vs_idx_map(op2data.symm)
    
    # Use transpose to convert to CSC format and exploit efficient row access
    non_red_T = SparseArrays.sparse(transpose(non_red))  # Transpose converts CSC to CSR (effectively). TODO: Define it directly in CSR format (:rows as last argument) here by merging the two functions and having access to the inverseMaps. This allocates new data, so it might be better to calculate this in the caller function.
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
        
        colbool, rowbool = compute_bipartite_vertex_cover(bigraph, algo)
    end

    # Find selected rows and columns based on the vertex cover
    row_select = findall(rowbool)
    # Sort row_select by how many columns each row covers, largest cover first
    # This helps optimize the MPO bond dimension
    sort!(row_select, by=row -> -length(nzrange(non_red_T, row)))

    col_select = findall(colbool)
    

    # Initialize output virtual spaces, tables, and factors
    virtSpaces_right_arr = []
    new_table = []
    new_factor = []
    site_entries = Vector{Tuple{Int, Int, Int, Float64}}() # Store site entries as (vs_left_idx, vs_right_idx, site_op_str_idx, factor)

    # Process selected rows first for better performance
    for row_idx in row_select
        # Create an output virtual space for this row
        vs_left_idx, site_op_idx = term_row[row_idx]

        # Get the list of possible right virtual spaces for this left index and site operator
        vs_left_tup = idx_vsQN_map[virtSpace_left_arr[vs_left_idx]]
        op_str = idx_localOps_map[site_op_idx]
        vs_right_list = [vsQN_idx_map[vs] for vs in keys(op2data.data[op_str][vs_left_tup])]
        
        # Since we transposed the sparse matrix, we get the columns as rows.
        matched_cols_range = nzrange(non_red_T, row_idx)
        
        # table_entry_n_matched_cols = length(matched_cols_range)
        table_entry_n_cols = length(term_col[1]) + 1

        for virtSpace_right in vs_right_list

            allowed_vs_right_list = [[vsQN_idx_map[vs] for vs in keys(op2data.data[idx_localOps_map[term_col[sparse_rows_T[matched_col_idx]][1]]])] for matched_col_idx in matched_cols_range]
            
            allowed_matched_idxs = findall(vs_allowed_by_col -> virtSpace_right in vs_allowed_by_col, allowed_vs_right_list)
            if isempty(allowed_matched_idxs)
                continue # Skip this right virtual space if it is not allowed by any column
            end
            
            allowed_matched_col_idxs = sparse_rows_T[matched_cols_range[allowed_matched_idxs]]
            table_entry_n_matched_cols = length(allowed_matched_col_idxs)
            
            push!(virtSpaces_right_arr, virtSpace_right)
            vs_right_idx = length(virtSpaces_right_arr)
            
            entry = (vs_left_idx, vs_right_idx, site_op_idx, 1.0)
            push!(site_entries, entry)
    
            table_entry = Matrix{Int}(undef, table_entry_n_matched_cols, table_entry_n_cols)
            for (i, col_idx) in enumerate(allowed_matched_col_idxs)
                table_entry[i, 1] = vs_right_idx
                table_entry[i, 2:table_entry_n_cols] = term_col[col_idx]
            end

            push!(new_table, table_entry)
            append!(new_factor, factors[sparse_vals_T[matched_cols_range[allowed_matched_idxs]]])

        end

        sparse_vals_T[matched_cols_range] .= 0
                
    end

    
    SparseArrays.dropzeros!(non_red_T)
    non_red = SparseArrays.sparse(non_red_T')
    
    # Process selected columns
    for col_idx in col_select
        # Create a multi-operator entry for this column
        
        next_op_str = idx_localOps_map[term_col[col_idx][1]]
        allowed_vs_right = keys(op2data.data[next_op_str])

        vs_right_idx_dict = Dict{Int, Int}()
        
        for (row_idx, val) in zip(findnz(non_red[:, col_idx])...)
            vs_left_idx, site_op_idx = term_row[row_idx]

            vs_left_tup = idx_vsQN_map[virtSpace_left_arr[vs_left_idx]]
            op_str = idx_localOps_map[site_op_idx]

            if !haskey(op2data.data[op_str], vs_left_tup)
                error("Operator $op_str does not have a mapping for virtual space $vs_left_tup. val= $(factors[val])")
            end
            vs_right_list = keys(op2data.data[op_str][vs_left_tup])

            # Filter vs_right_list to only include allowed virtual spaces
            vs_idx_right_list = [vsQN_idx_map[vs] for vs in vs_right_list if vs in allowed_vs_right]
            
            for virtSpace_right in vs_idx_right_list
                if haskey(vs_right_idx_dict, virtSpace_right)
                    vs_right_offset = vs_right_idx_dict[virtSpace_right]
                else
                    push!(virtSpaces_right_arr, virtSpace_right)
                    vs_right_offset = length(virtSpaces_right_arr)
                    vs_right_idx_dict[virtSpace_right] = vs_right_offset
                end
                
                entry = (vs_left_idx, vs_right_offset, site_op_idx, factors[val])
                push!(site_entries, entry)
            end
            
        end
        for vs_right_idx in values(vs_right_idx_dict)
            # Add the entry for this column
            push!(new_table, reshape(vcat([vs_right_idx], term_col[col_idx]), 1, :))
            push!(new_factor, 1.0)
        end
    end
    
    table = vcat(new_table...)

    @assert size(table, 1) == length(new_factor) "Table length ($(length(table))) does not match factor length ($(length(new_factor)))"

    return site_entries, virtSpaces_right_arr, table, new_factor

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
    compose_symbolic_site_sparse(site_entries, virtSpace_left, virtSpace_right, idx_localOps_map)

Create a sparse matrix representation of a quantum operator that maps between input and output virtual spaces.

This function composes a sparse matrix representation where each non-zero entry contains a collection 
of site operators. It is the sparse implementation counterpart to the dense operator composition function.

# Arguments
  - `site_entries::Vector{Tuple{Int, Int, Int, Float64}}`: A vector of tuples where each tuple contains:
  - `virtSpace_left`: The left virtual space indices.
  - `virtSpace_right`: The right virtual space indices.
  - `idx_localOps_map`: A mapping from operator indices to operator strings.

# Returns
- A sparse matrix of dimensions `(length(virtSpace_in), length(virtSpace_out))` where non-zero entries 
  contain tuples of operator string with their appropriate factors.

# Note
The function tracks non-zero entries using explicit coordinate lists (COO format) and
groups operators that map between the same input and output states.
"""


function compose_symbolic_site_sparse(site_entries, virtSpace_left, virtSpace_right, idx_localOps_map)
    # Use a dictionary to accumulate entries
    # Key: (row, col), Value: Vector of operators
    entries = Dict{Tuple{Int,Int}, Vector{Tuple{String, Float64}}}()
    
    # For each output operator
    for (vs_left_idx, vs_right_idx, op_idx, factor) in site_entries
        # For each composed operator in this output operator
        position = (vs_left_idx, vs_right_idx)
        
        # Create or retrieve the vector at this position
        if !haskey(entries, position)
            entries[position] = Tuple{String, Float64}[]
        end
        
        # Add the operator
        push!(entries[position], (idx_localOps_map[op_idx], factor))
    end
    
    # Create a sparse matrix from the dictionary
    I = Int[]
    J = Int[]
    V = Vector{Tuple{String, Float64}}[]
    
    for ((i, j), ops) in entries
        push!(I, i)
        push!(J, j)
        push!(V, ops)
    end
    
    return SparseArrays.sparse(I, J, V, length(virtSpace_left), length(virtSpace_right))
end
