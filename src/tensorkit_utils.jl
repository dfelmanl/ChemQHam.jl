using BlockTensorKit: ⊕

function get_qn_type(symm::String)
    """
    Get the quantum number type based on the symmetry type string representation.
    
    Parameters:
    - symm::String: String representation of the symmetry type
    
    Returns:
    - The quantum number type as a Tuple
    """
    if symm == "(FermionParity ⊠ Irrep[U₁])" || uppercase(symm) == "U1"
        return Tuple{Bool, Int}
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])" || uppercase(symm) == "U1U1"
        return Tuple{Bool, Int, Int}
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])" || uppercase(symm) == "U1SU2"
        return Tuple{Bool, Int, Rational{Int64}}
    else
        error("Unsupported quantum number representation: $symm")
    end
end

function get_QN_mapping_and_vs_multiplicity(virt_spaces_U1U1, qn_parser; symm="U1SU2")
    if symm == "(FermionParity ⊠ Irrep[U₁])" || uppercase(symm) == "U1"
        error("get_QN_mapping_and_vs_multiplicity for (FermionParity ⊠ Irrep[U₁]) has not been implemented yet.")
        return get_QN_mapping_and_vs_multiplicity_non_spin_symmetric(virt_spaces_U1U1, qn_parser)
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])" || uppercase(symm) == "U1U1"
        return get_QN_mapping_and_vs_multiplicity_non_spin_symmetric(virt_spaces_U1U1, qn_parser)
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])" || uppercase(symm) == "U1SU2"
        return get_symmQN_mapping_and_vs_SU2(virt_spaces_U1U1, qn_parser)
    else
        error("Unsupported quantum number representation: $symm")
    end
end


function get_qn_vs_maps__qn_mult_counts_U1SU2(virt_space, qn_parser)
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
            totSpin_fused = rationalize(abs(totSpin_fused)) # check if it's always 0
            qn = (fZ_fused, eCnt_fused, totSpin_fused)
            qn_mult_cnt[qn] += 1
            col_to_qn_map[i_col] = [(qn, 1, qn_mult_cnt[qn])]
        end
    end

    return col_to_qn_map, convert(Dict, qn_mult_cnt)
end

function get_symmQN_mapping_and_vs_SU2(virt_spaces_U1U1, qn_parser)
    qn_vs_maps__qn_mult_counts = [get_qn_vs_maps__qn_mult_counts_U1SU2(vsU1U1, qn_parser) for vsU1U1 in virt_spaces_U1U1]
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



function construct_empty_dense_mpo_site(phySpace::GradedSpace, left_qn_mult::Dict{QN, Int64}, right_qn_mult::Dict{QN, Int64}; dataType::DataType=Float64) where QN
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
    virtSpace_left = reduce(TensorKit.oplus, [qn_space(qn => cnt) for (qn, cnt) in left_qn_mult])
    virtSpace_right = reduce(TensorKit.oplus, [qn_space(qn => cnt) for (qn, cnt) in right_qn_mult])
    
    # Create a TensorMap for this site
    return zeros(dataType, virtSpace_left ⊗ phySpace ← virtSpace_right ⊗ phySpace)
end

function construct_empty_sparse_block_site(phySumSpace::SumSpace, left_vs::Vector{QN}, right_vs::Vector{QN}) where QN
    """
    Construct an empty sparse block MPO site with the given quantum number multiplicities.
    
    Parameters:
    - phySpace: Physical space (GradedSpace) for TensorKit tensors
    - left_qn_mult: Multiplicity count for each left quantum number
    - right_qn_mult: Multiplicity count for each right quantum number
    - dataType: Data type for tensor elements
    
    Returns:
    - empty_block_tensor: An empty sparse block MPO site represented as a SparseBlockTensorMap
    """

    # TODO: Add condition on symmetry
    sum_Vspace_left = ⊕([Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])](vs => 1) for vs in first.(left_vs)]...)
    sum_Vspace_right = ⊕([Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])](vs => 1) for vs in first.(right_vs)]...)

    block_space = sum_Vspace_left ⊗ phySumSpace ← sum_Vspace_right ⊗ phySumSpace
    empty_block_tensor = BlockTensorKit.spzeros(block_space)

    return empty_block_tensor
end

function get_offsetDict_and_SumSpaceVS(qn_mult_dict::Dict{QN, Int64}, symm_type::S) where {QN, S <: GradedSpace}
    # TODO: Remove if not used
    """
    Get the offset dictionary and flattened vector space for a given quantum number multiplicity dictionary.

    Parameters:
    - qn_mult_dict::Dict{QN, Int64}: Dictionary mapping quantum numbers to their multiplicities
    - symm_type::S: The symmetry type for the vector space

    Returns:
    - offset_dict::Dict{QN, Int64}: Dictionary mapping quantum numbers to their offsets
    - sum_space_vs::SumSpace: The flattened vector space representation as a `SumSpace` object
    """
    flatten_vs = Vector{S}()
    offset_dict = Dict{QN, Int64}()
    offset = 0
    for (qn, mult) in qn_mult_dict
        append!(flatten_vs, [symm_type(qn => 1) for _ in 1:mult])
        offset_dict[qn] = offset
        offset += mult
    end
    sum_space_vs = SumSpace(flatten_vs)
    return offset_dict, sum_space_vs
end

function get_full_virt_space(symm::String; as_dict::Bool=false)
    # TODO: Add to symmetry context
    if symm == "U1SU2"
        # Define the virtual spaces with an OrderedDict to assure that the trivial space ((0, 0, 0), 1) is always the first one
        vs_dict = OrderedDict{Tuple{Bool, Int, Rational{Int}}, Int}(
                    (0, 0, 0)=>2, 
                    (0, 0, 1)=>1, 
                    (1, 1, 1/2)=>2, 
                    (1, -1, 1/2)=>2, 
                    (0, 2, 0)=>1, 
                    (0, -2, 0)=>1, 
                    (0, 2, 1)=>1, 
                    (0, -2, 1)=>1
                )
    else
        throw(ArgumentError("Unsupported symmetry type: $symm"))
    end
    if as_dict
        return vs_dict
    else
        auxVecSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])](vs_dict...)
        return auxVecSpace
    end
end

function get_vs_idx_map(symm::String)
    # TODO: Add to symmetry context
    if symm == "U1SU2"
        vs_dict = get_full_virt_space(symm, as_dict=true)
        vs_idx_map = Dict{Tuple{Tuple{Bool, Int, Rational{Int64}}, Int64}, Int64}()
        idx = 1
        for (qn, mult) in vs_dict
            for i in 1:mult
                vs_idx_map[(qn, i)] = idx
                idx += 1
            end
        end

        return vs_idx_map
    else
        throw(ArgumentError("Unsupported symmetry type: $symm"))
    end
end

function GetCrAnLocalOpsU1U1(; dataType::DataType=Float64, spin_symm::Bool=false)
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

    phySpace = genPhySpace("U1U1")

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

function GetCrAnLocalOpsU1SU2(; dataType::DataType=Float64)
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
    
    symm = "U1SU2"
    # New Convention + 3rd multiplicity on (0,0,0) to even the probabilities when constracting on the same site
    auxVecSpace = get_full_virt_space(symm)

    
    # CREATION OPERATOR

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
    
    phySpace = genPhySpace(symm)
    ftree_type = FusionTree{sectortype(typeof(phySpace))}
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
        Float64
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

function get_all_local_ops_str(symm::String)
    """
    Returns a Vector of all local operator strings for the given symmetry.

    Parameters:
    - `symm`: A String representing the symmetry.
    Returns:
    - `all_local_ops_str`: A Vector of Strings representing all local operator strings.
    """
    all_local_ops_str = ["I"]
    if symm == "U1SU2"
        max_op_str_vector = ["a1", "a2", "c2", "c1"]
        n = length(max_op_str_vector)
        for i in 1:n
            for c in combinations(max_op_str_vector, i)
                op_str = join(c)
                push!(all_local_ops_str, op_str)
            end
        end
    else
        throw(ArgumentError("Unsupported symmetry: $symm. Supported symmetries are: 'U1SU2'."))
    end
    return all_local_ops_str
end


struct Op2Data{T}
    data::T
    ops::AbstractLocalOps{Float64}
    symm::String
    is_filled::Dict{String, Bool}
    
    function Op2Data(symm::String; fill_data::Bool=true)
        symm = validate_symmetry(symm)
        
        if symm == "U1U1"
            ops = GetCrAnLocalOpsU1U1(spin_symm=false)
        elseif symm == "U1SU2"
            # spin_symm is irrelevant, as SU2 is inherently spin symmetric. Add a warning?
            ops = GetCrAnLocalOpsU1SU2()
            ops = LocalOps_DoubleV(ops)
        else
            throw(ArgumentError("Unsupported symmetry: $symm. Supported symmetries are: 'U1U1', 'U1SU2'."))
        end

        all_local_ops = get_all_local_ops_str(symm)

        QNType = get_qn_type(symm)

        # Dictionary to store the sparse operator data
        OpDataDict = Dict{String, Dict{Tuple{QNType, Int}, Dict{Tuple{QNType, Int}, Vector{FusionTreeDataType(QNType)}}}}

        # Fill the operator data dictionary if requested
        if fill_data
            data = OpDataDict()
            for op_str in all_local_ops
                op_data = get_op_data(ops, op_str, QNType)
                data[op_str] = op_data
            end
            is_filled = Dict(op => true for op in all_local_ops)

        else
            data = OpDataDict()
            is_filled = Dict(op => false for op in all_local_ops)
        end

        return new{typeof(data)}(data, ops, symm, is_filled)
    end
end

function Base.getindex(op2data::Op2Data, input::Tuple{String, QN, Int, QN, Int}) where QN
    op_str, vs_out, vs_out_mult, vs_in, vs_in_mult = input

    if haskey(op2data.data, op_str)
        if haskey(op2data.data[op_str], (vs_out, vs_out_mult))
            if haskey(op2data.data[op_str][(vs_out, vs_out_mult)], (vs_in, vs_in_mult))
                
                return op2data.data[op_str][(vs_out, vs_out_mult)][(vs_in, vs_in_mult)]
                
            else
                @warn "No data found for operator $op_str with left virtual space $vs_out and multiplicity $vs_out_mult in the dictionary for right virtual space $vs_in and multiplicity $vs_in_mult."
                return [] # Return an empty vector if the data not found
            end
        else
            @warn "Operator $op_str with left virtual space $vs_out and multiplicity $vs_out_mult not found in the dictionary."
            return [] # Return an empty vector if the data not found
        end
    else
        
        op_data = get_op_data(op2data.ops, op_str, QN)

        if !haskey(op_data[op_str], (vs_out, vs_out_mult)) || !haskey(op_data[op_str][(vs_out, vs_out_mult)], (vs_in, vs_in_mult))
            @warn "No data found for operator $op_str with left virtual space $vs_out and multiplicity $vs_out_mult in the dictionary for right virtual space $vs_in and multiplicity $vs_in_mult."
        end
        op2data.data[op_str] = op_data
        op2data.is_filled[op_str] = true

        return op2data.data[op_str][(vs_out, vs_out_mult)][(vs_in, vs_in_mult)]
    end
end

function get_op_data(ops::AbstractLocalOps, op_str::String, ::Type{QN}) where QN
    op_TM = construct_op_TensorMap(ops, op_str) # note QN is not inferred as `sectortype(op_TM)` because we don't want any dependency on TensorKit and define the virtual spaces types ourselves
    
    # Use DefaultDict to eliminate haskey checks
    op_data = DefaultDict{Tuple{QN, Int}, DefaultDict{Tuple{QN, Int}, Vector{FusionTreeDataType(QN)}}}(
        () -> DefaultDict{Tuple{QN, Int}, Vector{FusionTreeDataType(QN)}}(
            () -> Vector{FusionTreeDataType(QN)}()
        )
    )
    
    for (f1, f2) in fusiontrees(op_TM)
        val = Matrix(op_TM[f1,f2][:,1,:,1])
        if !iszero(val) # !all(val .== 0)
            vs_out, ftree_left = ftree_data(f1)
            vs_in, ftree_right = ftree_data(f2)


            for (vs_out_mult, vs_in_mult, nzval) in zip(findnz(SparseArrays.sparse(val))...)
                push!(op_data[(vs_out, vs_out_mult)][(vs_in, vs_in_mult)], (ftree_left, ftree_right, nzval))
            end
        end
    end
    
    return op_data
end

function construct_op_TensorMap(ops::AbstractLocalOps, op_str::String)
    """
    Construct a TensorMap for a given operator string using the Op2Data structure.
    
    Parameters:
    - ops: AbstractLocalOps containing the operator definitions
    - op_str: String representing the operator
    
    Returns:
    - TensorMap representing the operator
    """
    if op_str == "I"
        # Identity operator
        op_TM = ops["I"]
    else
        op_TM = reduce(*, [ops[join(collect(op_str)[i:i+1])] for i in reverse(1:2:length(op_str))])
    end
    
    return op_TM
end



function fill_mpo_site_SU2!(sblock_site, symb_mpo_site, vs_left, vs_right, op2data, ftree_type, physical_abs_offsets; verbose=false)

    for (row, col, ops) in zip(findnz(symb_mpo_site)...)

        left_vsQN, left_vsMult = vs_left[row]
        right_qnQN, right_vsMult = vs_right[col]

        for (op_str, op_coef) in ops

            op_data = op2data[(op_str, left_vsQN, left_vsMult, right_qnQN, right_vsMult)]

            for (ftree_left, ftree_right, val) in op_data

                ps_out = ftree_left[1][2] # physical sector out
                ps_in = ftree_right[1][2] # physical sector in

                phy_l = physical_abs_offsets[ps_out]
                phy_r = physical_abs_offsets[ps_in]

                sp_coordinates = (row, phy_l, col, phy_r)

                ftree_tm = copy(sblock_site[sp_coordinates...])
                old_val = only(ftree_tm[ftree_type(ftree_left...), ftree_type(ftree_right...)])
                new_val = old_val + val * op_coef
                ftree_tm[ftree_type(ftree_left...), ftree_type(ftree_right...)] .= new_val

                verbose && println("Filling MPO site for op_str: $op_str, left QN: $left_vsQN, left_op_mult: $left_vsMult, right QN: $right_qnQN, right_op_mult: $right_vsMult")
                verbose && println("sp_coordinates: $sp_coordinates, old_val: $old_val, new_val: $new_val, CG coef val: $val, operator's coef: $op_coef")
                push!(sblock_site.data, CartesianIndex(sp_coordinates...) => ftree_tm)
            end
        end
    end

end

function fill_mpo_site_U1U1!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type; spin_symm::Bool=false, verbose::Bool=false)

    for (row, col, ops) in zip(findnz(symb_mpo_site)...)
        (left_qn, left_mult) = vs_map_left[row]
        (right_qn, right_mult) = vs_map_right[col]
        
        for op in ops

            if spin_symm
                throw(NotImplementedError("Spin symmetry is not implemented for U1U1 operators. Choose spin_symm=false."))
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

                verbose && println("Filling MPO site for operator: $op, op_str $op_str, left QN: $left_qn, right QN: $right_qn, left_mult: $left_mult, right_mult: $right_mult, ftree_left: $ftree_left, ftree_right: $ftree_right, value: $val, coef: $(op.coefficient)")
                mpo_site[f1,f2][left_mult,:,right_mult,:] .+= val * op.coefficient
                
            end
        end
    end

end

function validate_symmetry(symm)
    if uppercase(symm) == "U1U1"
        return "U1U1"
    elseif uppercase(symm) == "U1SU2"
        return "U1SU2"
    else
        throw(ArgumentError("Unsupported symmetry type: $symm. Supported types are 'U1U1' and 'U1SU2'."))
    end
end

symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, symm::String, vsQN_idx_map, op2data; kwargs...) = _symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, validate_symmetry(symm), vsQN_idx_map, op2data; kwargs...)

function _symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, symm::String, vsQN_idx_map, op2data; dataType::DataType=Float64, verbose=false)

    # TODO: Remove spin_symm parameter when the MPO construction is stable. It should always be false for U1U1 and true for U1SU2.
    
    mpo_sites = Vector{SparseBlockTensorMap}(undef, length(symbolic_mpo)) # Allocate TensorMap with the already known properties (type, shape, etc.)

    phySumSpace = genFlattenedPhySpace(symm)
    physical_abs_offsets = get_physical_abs_offsets(symm)

    ftree_type = FusionTree{sectortype(phySumSpace)}

    for (isite, symb_mpo_site) in enumerate(symbolic_mpo)
        verbose && println("Processing MPO site $isite with symbolic data: $symb_mpo_site")
        vs_left = virt_spaces[isite]
        vs_right = virt_spaces[isite+1]
        
        sblock_site = construct_empty_sparse_block_site(phySumSpace, vs_left, vs_right)
        
        if symm == "U1SU2"
            fill_mpo_site_SU2!(sblock_site, symb_mpo_site, vs_left, vs_right, op2data, ftree_type, physical_abs_offsets; verbose=verbose)
        elseif symm == "U1U1"
            fill_mpo_site_U1U1!(sblock_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type; verbose=verbose)
        else
            throw(ArgumentError("Unsupported symmetry type: $symm"))
        end
        
        mpo_sites[isite] = sblock_site
    end

    return mpo_sites
end


function mpo_to_mat(H_mpo)
    N_spt = length(H_mpo)
    symm_space = TensorKit.space(H_mpo[1], 1)
    
    # Create boundary tensors with zero total quantum numbers
    boundaryMPO_L = TensorMap(ones, one(symm_space), oneunit(symm_space))
    boundaryMPO_R = TensorMap(ones, oneunit(symm_space), one(symm_space))
    
    H_contraction = boundaryMPO_L
    
    # Contract with remaining MPO tensors sequentially
    for i in 1:N_spt
        domain_legs = tuple(vcat(2*i+1, 1:(2*i-1))...)
        codomain_leg = (2*i,)
        
        H_contraction = TensorKit.permute(H_contraction * TensorKit.permute(H_mpo[i], (1,), (2,3,4)), 
                             domain_legs, codomain_leg)
    end
    
    right_legs = collect(N_spt+1:2*N_spt)
    left_legs = collect(N_spt:-1:1)
    
    H_contraction = TensorKit.permute(H_contraction * boundaryMPO_R, tuple(right_legs...), tuple(left_legs...))
    
    # Convert to TensorMap when working with `SparseBlockTensorMap`s
    if !isa(H_contraction, TensorMap)
        H_contraction = TensorMap(H_contraction)
    end

    # Reshape to matrix form
    mpo_mat = SparseArrays.sparse(reshape(convert(Array, H_contraction), (4^N_spt, 4^N_spt)))
    return mpo_mat
end

"""
    genPhysSpace(symm)

Generate the physical space based on the given symmetry type `symm`.

# Arguments
- `symm::String`: A string representing the type of physical space to generate. The current options (case insensitive) are: 'U1', 'Par' for (U1 ⊠ fermionicParity), and 'SU2' for (U1 ⊠ SU2 ⊠ fermionicParity).

# Returns
- A physical space object corresponding to the specified `symm`.
"""
function genPhySpace(symm)
    # FermionParity must always be imposed
    
    if uppercase(symm) == "U1"
        phySpace = Vect[(FermionParity ⊠ Irrep[U₁])]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1)
    elseif uppercase(symm) == "U1U1"
        # (fParity, total count, spin count)
        phySpace = Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1, (0, 2, 0) => 1)
    elseif uppercase(symm) == "U1SU2"
        phySpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (0, 2, 0) => 1)
    else
        throw(ArgumentError("The specified symmetry type '$symm' is not implemented. The current options are: 'U1', 'U1U1', and 'U1SU2'"))
    end

    return phySpace

end

function genFlattenedPhySpace(symm)
    # This sets the convention for the order of each physical space sector.
    
    if uppercase(symm) == "U1"
        phySumSpace = Vect[(FermionParity ⊠ Irrep[U₁])]((0, 0) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁])]((1, 1) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁])]((1, 1) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁])]((0, 2) => 1)
    elseif uppercase(symm) == "U1U1"
        # (fParity, total count, spin count)
        phySumSpace = Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((0, 0, 0) => 1) ⊕ Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((1, 1, 1 // 2) => 1) ⊕ Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((1, 1, -1 // 2) => 1) ⊕ Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((0, 2, 0) => 1)
    elseif uppercase(symm) == "U1SU2"
        phySumSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((1, 1, 1 // 2) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 2, 0) => 1)
    else
        throw(ArgumentError("The specified symmetry type '$symm' is not implemented. The current options are: 'U1', 'U1U1', and 'U1SU2'"))
    end

    return phySumSpace

end

function get_physical_abs_offsets(symm::String)
    """
    Returns a function that maps the physical quantum numbers to their absolute offsets in the flattened physical space.
    """
    phySumSpace = genFlattenedPhySpace(symm)
    physical_offsets = Dict(ftree_inner_data(phySumSpace[i].dims.keys[1])  => i for i in 1:length(phySumSpace))
    
    return physical_offsets
end
