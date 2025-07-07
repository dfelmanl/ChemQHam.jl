import TensorKit: ⊕

"""
    QNParser(phySpace::GradedSpace)

Convert a `GradedSpace` to an appropriate quantum number parser function.
Returns a function that can parse spin-up and spin-down occupation numbers into the appropriate quantum number representation.

# Arguments
- `phySpace::GradedSpace`: The graded space defining the symmetry type

# Returns
- A function that converts spin occupation numbers to quantum numbers
"""
QNParser(phySpace::GradedSpace) = QNParser(TensorKit.type_repr(sectortype(phySpace)))

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
function QNParser(symm::String)
    if symm == "(FermionParity ⊠ Irrep[U₁])" || uppercase(symm) == "U1"
        return fNumberQNParser
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])" || uppercase(symm) == "U1U1"
        return U1U1QNParser
    elseif symm == "(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])" || uppercase(symm) == "U1SU2"
        return SU2doubleQNParser
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
@addSpinIteratorCapability SU2doubleQNParser

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
    # auxVecSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0)=>2, (0, 0, 1)=>1, (1, 1, 1/2)=>1, (1, -1, 1/2)=>1, (0, 2, 0)=>1, (0, -2, 0)=>1, (0, 2, 1)=>1, (0, -2, 1)=>1)
    
    # New Convention + 3rd multiplicity on (0,0,0) to even the probabilities when constracting on the same site
    auxVecSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0)=>2, (0, 0, 1)=>1, (1, 1, 1/2)=>2, (1, -1, 1/2)=>2, (0, 2, 0)=>1, (0, -2, 0)=>1, (0, 2, 1)=>1, (0, -2, 1)=>1)

    
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
    
    phySpace = genPhySpace("U1SU2")
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
    
    function Op2Data(symm::String)
        
        if symm == "U1U1"
            ops = GetCrAnLocalOpsU1U1(spin_symm=false)
        elseif symm == "U1SU2"
            # spin_symm is irrelevant, as SU2 is inherently spin symmetric. Add a warning?
            ops = GetCrAnLocalOpsU1SU2()
            ops = LocalOps_DoubleV(ops)
        else
            throw(ArgumentError("Unsupported symmetry: $symm. Supported symmetries are: 'U1U1', 'U1SU2'."))
        end

        # Define the OpDataDict type
        QNType = get_qn_type(symm)
        OpDataDict = Dict{String, Dict{Tuple{QNType, QNType}, Vector{FusionTreeDataType(QNType)}}}

        return new{OpDataDict}(OpDataDict(), ops, symm)
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
    if op2data.symm == "U1SU2"
        op_TM = reduce(*, [op2data.ops[join(collect(op_str)[i:i+1])] for i in reverse(1:2:length(op_str))])
    elseif op2data.symm == "U1U1"
        # This assumes GetCrAnLocalOpsU1U1(spin_symm=false), as the operators are defined with a spin character as "c↑", "c↓", "a↑", "a↓"
        op_TM = reduce(*, [op2data.ops[join(collect(op_str)[i:i+1])] for i in reverse(1:2:length(op_str))])
    else
        throw(ArgumentError("Unsupported symmetry: $(op2data.symm)"))
    end
    
    return op_TM
end



function fill_mpo_site_SU2!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type; verbose=false)

    for (row, col, ops) in zip(findnz(symb_mpo_site)...)

        for (left_qn, left_op_mult, left_MPOsite_mult) in vs_map_left[row]
            for (right_qn, right_op_mult, right_MPOsite_mult) in vs_map_right[col]
        
                for op in ops

                    op_str = op.operator

                    op_data = op2data[(op_str, left_qn, right_qn)]

                    for (ftree_left, ftree_right, val) in op_data
                        f1 = ftree_type(ftree_left...)
                        f2 = ftree_type(ftree_right...)

                        verbose && println("Filling MPO site for operator: $op, op_str $op_str, left QN: $left_qn, left_op_mult: $left_op_mult, right QN: $right_qn, right_op_mult: $right_op_mult left_mult: $left_MPOsite_mult, right_mult: $right_MPOsite_mult, ftree_left: $ftree_left, ftree_right: $ftree_right, val: $val, val[left_op_mult, right_op_mult]: $(val[left_op_mult, right_op_mult]),  coef: $(op.coefficient)")
                        mpo_site[f1,f2][left_MPOsite_mult, 1, right_MPOsite_mult, 1] += val[left_op_mult, right_op_mult] * op.coefficient
                    end
                end
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

symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, symm::String; kwargs...) = _symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, validate_symmetry(symm); kwargs...)

function _symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, symm::String; dataType::DataType=Float64, verbose=false)
    # TODO: Remove spin_symm parameter when the MPO construction is stable. It should always be false for U1U1 and true for U1SU2.
    
    mpo_sites = Vector{TensorMap}(undef, length(symbolic_mpo)) # Allocate TensorMap with the already known properties (type, shape, etc.)
    qn_parser = QNParser(symm)

    qn_vs_maps, qn_mult_counts = get_QN_mapping_and_vs_multiplicity(virt_spaces, qn_parser; symm=symm)

    op2data = Op2Data(symm)
    phySpace = genPhySpace(symm)
    ftree_type = FusionTree{sectortype(phySpace)}

    for (isite, symb_mpo_site) in enumerate(symbolic_mpo)
        qn_mult_counts_left = qn_mult_counts[isite]
        qn_mult_counts_right = qn_mult_counts[isite+1]
        mpo_site = construct_empty_mpo_site(phySpace, qn_mult_counts_left, qn_mult_counts_right; dataType=dataType)
        
        vs_map_left = qn_vs_maps[isite]
        vs_map_right = qn_vs_maps[isite+1]
        
        if symm == "U1SU2"
            fill_mpo_site_SU2!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type; verbose=verbose)
        elseif symm == "U1U1"
            fill_mpo_site_U1U1!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, op2data, ftree_type; verbose=verbose)
        else
            throw(ArgumentError("Unsupported symmetry type: $symm"))
        end
        
        mpo_sites[isite] = mpo_site
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
    
    # Output physical legs: N_spt+1:2*N_spt
    # Input physical legs in reverse order: N_spt:-1:1
    right_legs = collect(N_spt+1:2*N_spt)
    left_legs = collect(N_spt:-1:1)
    
    H_contraction = TensorKit.permute(H_contraction * boundaryMPO_R, tuple(right_legs...), tuple(left_legs...))
    
    # Reshape to matrix form
    mpo_mat = sparse(reshape(convert(Array, H_contraction), (4^N_spt, 4^N_spt)))
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
        orbPhysSpace = Vect[(FermionParity ⊠ Irrep[U₁])]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1)
    elseif uppercase(symm) == "U1U1"
        # (fParity, total count, spin count)
        orbPhysSpace = Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1, (0, 2, 0) => 1)
    elseif uppercase(symm) == "U1SU2"
        orbPhysSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (0, 2, 0) => 1)
    else
        throw(ArgumentError("The specified symmetry type '$symm' is not implemented. The current options are: 'U1', 'U1U1', and 'U1SU2'"))
    end

    return orbPhysSpace

end