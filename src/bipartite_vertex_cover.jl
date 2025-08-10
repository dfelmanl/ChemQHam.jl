
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
    matchingU = Vector{Union{Int,Nothing}}(nothing, nU)

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
    hungarian_max_bipartite_matching(bigraph)

Implement the Hungarian algorithm for bipartite matching, which finds an optimal
assignment between two sets of nodes in a bipartite graph.

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
    match = Vector{Union{Int,Nothing}}(undef, nV)
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
    for u = 1:nU
        augment(u, fill(false, nV))
    end

    return match
end


"""
    compute_bipartite_vertex_cover(bigraph, algo)

Compute a minimum vertex cover of a bipartite graph using the specified algorithm.

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
