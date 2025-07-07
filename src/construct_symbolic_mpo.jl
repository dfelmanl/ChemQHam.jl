
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
function construct_symbolic_mpo(table, primary_ops, factor; algo="Hungarian", verbose=false)

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

function construct_symbolic_mpo(op_terms::ChemOpSum; kwargs...)
    table, primary_ops, factors = terms_to_table(op_terms)
    return construct_symbolic_mpo(table, primary_ops, factors; kwargs...)
end

function construct_symbolic_mpo(chem_data, ord; ops_tol=1e-14, maxdim=2^30, algo="Hungarian", spin_symm::Bool=false, verbose=false)
    # Generate the ChemOpSum object from the Hamiltonian coefficients:
    terms = gen_ChemOpSum(chem_data, ord; tol=ops_tol, spin_symm=spin_symm)

    # Convert to operator table
    table, primary_ops, factors = _terms_to_table(chem_data.N_spt, terms)
    
    # Build symbolic MPO with graph-based optimization
    symbolic_mpo, virt_spaces = construct_symbolic_mpo(table, primary_ops, factors; algo=algo, verbose=verbose)
    
    return symbolic_mpo, virt_spaces
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
    
    virtSpaces_out, table, new_factor = _decompose_graph(term_row, term_col, non_red, virtSpace_in, factor, primary_ops, algo)
    
    return virtSpaces_out, table, new_factor
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
