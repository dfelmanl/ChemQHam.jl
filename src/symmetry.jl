# Symmetry context system for ChemQHam
# This file defines the symmetry context that holds all symmetry-specific information

"""
    AbstractSymmetryContext

Abstract base type for symmetry contexts. All concrete symmetry contexts should inherit from this type.
Note that it is always implied that the fermionic symmetry is included, as it is required for the electronic behavior in quantum chemistry.
"""
abstract type AbstractSymmetryContext end

# Abstract base type for local operators
abstract type AbstractLocalOps{T} end

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

########################################################################
################## U1SU2 Specific Symmetry functions ###################
########################################################################

"""
    U1SU2SymmetryContext

Symmetry context for `fZ ⊠ U1 ⊠ SU2` symmetry, commonly used in quantum chemistry.
This context pre-computes and stores all necessary mappings and data structures
needed for symbolic MPO construction.
"""
struct U1SU2SymmetryContext{qn_type} <: AbstractSymmetryContext
    name::String
    qn_type::DataType
    is_spin_symm::Bool
    operators::AbstractLocalOps{Float64}
    operator_data::Dict{String, Dict{Tuple{qn_type, Int}, Dict{Tuple{qn_type, Int}, Vector{FusionTreeDataType(qn_type)}}}}
    local_ops_idx_map::Dict{String, Int}
    idx_local_ops_map::Dict{Int, String}
    vsQN_idx_map::Dict{Tuple{qn_type, Int}, Int}
    idx_vsQN_map::Dict{Int, Tuple{qn_type, Int}}
    all_local_ops::Vector{String}
    is_filled::Dict{String, Bool}
    
    function U1SU2SymmetryContext(; fill_data::Bool=true)
        name = "U1SU2"
        qn_type = Tuple{Bool, Int, Rational{Int}}
        is_spin_symm = true
        
        # Initialize operators
        ops = get_cr_an_local_ops_U1SU2()
        ops = LocalOps_SpinSymm(ops)
        
        # Get all local operator strings
        all_local_ops = get_all_local_ops_str(name)
        
        # Create local operators index mapping
        local_ops_idx_map = Dict(op => i for (i, op) in enumerate(all_local_ops))
        idx_local_ops_map = Dict(v => k for (k, v) in local_ops_idx_map)
        
        # Create virtual space index mapping
        vsQN_idx_map = get_virt_space_idx_map_U1SU2()
        idx_vsQN_map = Dict(v => k for (k, v) in vsQN_idx_map)
        
        # Initialize operator data dictionary
        OpDataDict = Dict{String, Dict{Tuple{qn_type, Int}, Dict{Tuple{qn_type, Int}, Vector{FusionTreeDataType(qn_type)}}}}
        
        # Fill the operator data dictionary if requested
        if fill_data
            operator_data = OpDataDict()
            for op_str in all_local_ops
                op_data = get_op_data(ops, op_str, qn_type)
                operator_data[op_str] = op_data
            end
            is_filled = Dict(op => true for op in all_local_ops)
        else
            operator_data = OpDataDict()
            is_filled = Dict(op => false for op in all_local_ops)
        end
        
        return new{qn_type}(name, qn_type, is_spin_symm, ops, operator_data, local_ops_idx_map, idx_local_ops_map, vsQN_idx_map, idx_vsQN_map, all_local_ops, is_filled)
    end
end

# Helper function to compute virtual space index mapping for U1SU2
function get_virt_space_idx_map_U1SU2()
    vs_dict = get_full_virt_space_multiplicities_U1SU2()
    vsQN_idx_map = Dict{Tuple{Tuple{Bool, Int, Rational{Int}}, Int}, Int}()
    idx = 1
    for (qn, mult) in vs_dict
        for i in 1:mult
            vsQN_idx_map[(qn, i)] = idx
            idx += 1
        end
    end
    return vsQN_idx_map
end

function get_full_virt_space_multiplicities_U1SU2()
    # IMPORTANT: the trivial space ((0, 0, 0), 1) must be mapped to 1, so we make use of OrderedDict to assure that 
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
    return vs_dict
end

# Define the U1SU2 local operators
function get_cr_an_local_ops_U1SU2(; dataType::DataType=Float64)
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
    
    # New Convention + 3rd multiplicity on (0,0,0) to even the probabilities when constracting on the same site
    virt_space_dict = get_full_virt_space_multiplicities_U1SU2()
    auxVecSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])](virt_space_dict...)

    
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
    
    phySpace = genPhySpace("U1SU2")
    ftree_type = FusionTree{sectortype(typeof(phySpace))}
    crOp = zeros(dataType, auxVecSpace ⊗ phySpace, auxVecSpace ⊗ phySpace)
    # TODO: Check how can this be inherently defined. Maybe this forces the convention to be changed
    for (f1, f2, ftree_array) in crOp_ftree_nzdata

        vs_right = f2[1][1]

        if vs_right == (0, 0, 1.0)
            # Flip sign to the first col (0,0,1)_1 # Solves the addiitonal negative sign to the long cases of caac and acca by flipping signs to (0,0,1)_1 to An2 vs_right and Cr2 vs_left (or viceversa); in order to have anticommutivity in the (0,0,1)_1 virtual space, (+1, -1) with opposite signs
            ftree_array[:,1] *= -1
        elseif vs_right == (0, 0, 0.0)
            # Flip sign to the second col (0,0,0)_2 # Solves the addiitonal negative sign to the long cases of caac and acca by flipping signs to (0,0,1)_1 to An2 vs_right and Cr2 vs_left (or viceversa); in order to have anticommutivity in the (0,0,1)_1 virtual space, (+1, -1) with opposite signs
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

        vs_right = f2[1][1]

        if vs_right == (0, 0, 0.0)
            # Flip sign to the second column (0,0,0)_2
            ftree_array[:,2] = ftree_array[:,2] * -1
        elseif vs_right == (0, 0, 1.0)
            # Flip sign to the first column (0,0,1)_1
            ftree_array[:,1] *= -1
        elseif vs_right == (1, -1, 0.5)
            # Flip sign to the first column (1,+-1,0.5)_1
            ftree_array[:,1] *= -1
        elseif vs_right == (0, -2, 0.0)
            # Flip sign to the first column (0,-2,0)_1
            ftree_array[:,1] *= -1
        elseif vs_right == (0, -2, 1.0)
            # Flip sign to the first column (0,-2,0)_1
            ftree_array[:,1] *= -1
        end

        
        vs_left = f1[1][1]
        if vs_left == (0, 0, 1.0)
            # Flip sign to the first row (0,0,1)_1 # Solves the addiitonal negative sign to the long cases of caac and acca by flipping signs to (0,0,1)_1 to An2 vs_right and Cr2 vs_left (or viceversa); in order to have anticommutivity in the (0,0,1)_1 virtual space, (+1, -1) with opposite signs
            ftree_array[1,:] *= -1
        elseif vs_left == (0, 0, 0.0)
            # Flip sign to the second row (0,0,0)_1 # Solves the addiitonal negative sign to the long cases of caac and acca by flipping signs to (0,0,1)_1 to An2 vs_right and Cr2 vs_left (or viceversa); in order to have anticommutivity in the (0,0,1)_1 virtual space, (+1, -1) with opposite signs
            ftree_array[2,:] *= -1
        end
        
        
        anOp[ftree_type(f1...), ftree_type(f2...)][:,1,:,1] = ftree_array
    end

    # IDENTITY OPERATOR
    idOp = localIdOp(phySpace, auxVecSpace; dataType=dataType)
    
    return LocalOps(crOp, anOp, idOp)
end


########################################################################
################### U1U1 Specific Symmetry functions ###################
########################################################################

"""
    U1U1SymmetryContext

Symmetry context for `fZ ⊠ U1 ⊠ U1` symmetry (particle number and spin conservation).
This context pre-computes and stores all necessary mappings and data structures
needed for symbolic MPO construction.
"""
struct U1U1SymmetryContext{qn_type} <: AbstractSymmetryContext
    name::String
    qn_type::DataType
    is_spin_symm::Bool
    operators::AbstractLocalOps{Float64}
    operator_data::Dict{String, Dict{Tuple{qn_type, Int}, Dict{Tuple{qn_type, Int}, Vector{FusionTreeDataType(qn_type)}}}}
    local_ops_idx_map::Dict{String, Int}
    idx_local_ops_map::Dict{Int, String}
    vsQN_idx_map::Dict{Tuple{qn_type, Int}, Int}
    idx_vsQN_map::Dict{Int, Tuple{qn_type, Int}}
    all_local_ops::Vector{String}
    is_filled::Dict{String, Bool}
    
    function U1U1SymmetryContext(; fill_data::Bool=true)
        throw(NotImplementedError("U1U1 symmetry context is a WIP and is not yet implemented."))
        name = "U1U1"
        qn_type = Tuple{Bool, Int, Int}
        is_spin_symm = false
        
        # Initialize operators
        ops = get_cr_an_local_ops_U1U1(spin_symm=is_spin_symm)
        
        # Get all local operator strings
        all_local_ops = get_all_local_ops_str(name)
        
        # Create local operators index mapping
        local_ops_idx_map = Dict(op => i for (i, op) in enumerate(all_local_ops))
        idx_local_ops_map = Dict(v => k for (k, v) in local_ops_idx_map)

        # Create virtual space index mapping
        vsQN_idx_map = Dict{Any, Int}()  # Placeholder for now.
        idx_vsQN_map = Dict(v => k for (k, v) in vsQN_idx_map)

        # Initialize operator data dictionary
        OpDataDict = Dict{String, Dict{Tuple{qn_type, Int}, Dict{Tuple{qn_type, Int}, Vector{FusionTreeDataType(qn_type)}}}}
        
        # Fill the operator data dictionary if requested
        if fill_data
            operator_data = OpDataDict()
            for op_str in all_local_ops
                op_data = get_op_data(ops, op_str, qn_type)
                operator_data[op_str] = op_data
            end
            is_filled = Dict(op => true for op in all_local_ops)
        else
            operator_data = OpDataDict()
            is_filled = Dict(op => false for op in all_local_ops)
        end
        
        return new{qn_type}(name, qn_type, is_spin_symm, ops, operator_data, local_ops_idx_map, idx_local_ops_map, vsQN_idx_map, idx_vsQN_map, all_local_ops, is_filled)
    end
end

# Define the U1U1 local operators
function get_cr_an_local_ops_U1U1(; dataType::DataType=Float64, spin_symm::Bool=false)
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


########################################################################
################## General Symmetry Context Functions ##################
########################################################################

"""
    create_symmetry_context(symmetry_name::String; kwargs...)

Factory function to create the appropriate symmetry context based on the symmetry name.

# Arguments
- `symmetry_name::String`: Name of the symmetry ("U1SU2", "U1U1", etc.)
- `kwargs...`: Additional keyword arguments passed to the constructor

# Returns
- `AbstractSymmetryContext`: The appropriate symmetry context instance
"""
function create_symmetry_context(symmetry_name::String; kwargs...)
    symmetry_name = validate_symmetry(symmetry_name)
    
    if symmetry_name == "U1SU2"
        return U1SU2SymmetryContext(; kwargs...)
    elseif symmetry_name == "U1U1"
        return U1U1SymmetryContext(; kwargs...)
    else
        throw(ArgumentError("Unsupported symmetry: $symmetry_name. Supported symmetries are: 'U1SU2', 'U1U1'."))
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

function Base.getindex(symm_ctx::AbstractSymmetryContext, input::Tuple{String, QN, Int, QN, Int}) where QN
    op_str, vs_left, vs_left_mult, vs_right, vs_right_mult = input

    if haskey(symm_ctx.operator_data, op_str)
        if haskey(symm_ctx.operator_data[op_str], (vs_left, vs_left_mult))
            if haskey(symm_ctx.operator_data[op_str][(vs_left, vs_left_mult)], (vs_right, vs_right_mult))
                
                return symm_ctx.operator_data[op_str][(vs_left, vs_left_mult)][(vs_right, vs_right_mult)]
                
            else
                @warn "No data found for operator $op_str with left virtual space $vs_left and multiplicity $vs_left_mult in the dictionary for right virtual space $vs_right and multiplicity $vs_right_mult."
                return [] # Return an empty vector if the data not found
            end
        else
            @warn "Operator $op_str with left virtual space $vs_left and multiplicity $vs_left_mult not found in the dictionary."
            return [] # Return an empty vector if the data not found
        end
    else
        
        op_data = get_op_data(symm_ctx.operators, op_str, QN)

        if !haskey(op_data[op_str], (vs_left, vs_left_mult)) || !haskey(op_data[op_str][(vs_left, vs_left_mult)], (vs_right, vs_right_mult))
            @warn "No data found for operator $op_str with left virtual space $vs_left and multiplicity $vs_left_mult in the dictionary for right virtual space $vs_right and multiplicity $vs_right_mult."
        end
        symm_ctx.operator_data[op_str] = op_data
        symm_ctx.is_filled[op_str] = true

        return symm_ctx.operator_data[op_str][(vs_left, vs_left_mult)][(vs_right, vs_right_mult)]
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
        if !iszero(val)
            vs_left, ftree_left = ftree_data(f1)
            vs_right, ftree_right = ftree_data(f2)


            for (vs_left_mult, vs_right_mult, nzval) in zip(findnz(SparseArrays.sparse(val))...)
                push!(op_data[(vs_left, vs_left_mult)][(vs_right, vs_right_mult)], (ftree_left, ftree_right, nzval))
            end
        end
    end
    
    return op_data
end

function construct_op_TensorMap(ops::AbstractLocalOps, op_str::String)
    """
    Construct a TensorMap for a given operator string using the symmetry context.
    
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


function get_all_local_ops_str(symm::String)
    """
    Returns a Vector of all local operator strings for the given symmetry.

    Parameters:
    - `symm`: A String representing the symmetry.
    Returns:
    - `all_local_ops_str`: A Vector of Strings representing all local operator strings.
    """
    if symm == "U1SU2"
        all_local_ops_str = get_all_local_ops_str(["a1", "a2", "c2", "c1"])
    else
        # For non symmetric operator sums, one may have to take the union of the restuls from ["a↑", "a↑", "c↑", "c↑"], ["a↓", "a↓", "c↓", "c↓"], ["a↑", "a↓", "c↓", "c↑"] and ["a↓", "a↑", "c↑", "c↓"].
        throw(ArgumentError("Unsupported symmetry: $symm. Supported symmetries are: 'U1SU2'."))
    end
    return all_local_ops_str
end

function get_all_local_ops_str(max_op_str_vector::Vector{String})
    """
    Returns a Vector of all local operator strings for the given symmetry.

    Note that this function assumes that operator strings cannot be permuted:
    when the operator sum term is generated, it only counts each symmetric string once.
    i.e. [a1]_1 [a2]_2 [c2]_3 [c1]_4 is the same as [a2]_1 [a1]_2 [c1]_3 [c2]_4.

    Parameters:
    - `max_op_str_vector`: The longest possible operator string for a given symmetry at any site.
                           
    Returns:
    - `all_local_ops_str`: A Vector of Strings representing all local operator strings.
    """
    all_local_ops_str = ["I"]
    n = length(max_op_str_vector)
    for i in 1:n
        for c in combinations(max_op_str_vector, i)
            op_str = join(c)
            push!(all_local_ops_str, op_str)
        end
    end
    return all_local_ops_str
end

# Operator related functions
# Note that although these functions use TensorKit's backend, the input/output are independent from the backend.
# They are used to calculate the Glebsch-Gordan coefficients and other operator-related data, like allowed virtual spaces for a given operator.


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
    
    if uppercase(symm) == "U1U1"
        # (fParity, total count, spin count)
        phySpace = Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1, (0, 2, 0) => 1)
    elseif uppercase(symm) == "U1SU2"
        phySpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (0, 2, 0) => 1)
    else
        throw(ArgumentError("The specified symmetry type '$symm' is not implemented. The current options are: 'U1', 'U1U1', and 'U1SU2'"))
    end

    return phySpace

end

# General identity operator
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
            # push!(data, convert(Int, sector.isodd))
            push!(data, sector.isodd)
        elseif qn == :charge
            # push!(data, convert(Int, sector.charge))
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


# Define the local operators with spin symmetry, where they have to be expanded to include the `a2` and `c2` operators, including all the required fermionic anticommutations.
struct LocalOps_SpinSymm{T} <: AbstractLocalOps{T}
    base_ops::LocalOps{T}

    LocalOps_SpinSymm(base_ops::LocalOps{T}) where T = new{T}(base_ops)
end


function Base.getindex(lo2V::LocalOps_SpinSymm, key::String)
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
                vs_left = ftree_data(f1)[1]
                if iszero(vs_left[1])
                    # Swap columns (domain)
                    c[f1,f2][:,1,:,1] = ftree_array[:,1,[2,1],1]

                    if vs_left == (0, -2, 1.0)
                        # Flip sign to the first row (0,-2,1)_1
                        c[f1,f2][1,1,:,1] *= -1
                    end
                else
                    # Swap rows (codomain)
                    c[f1,f2][:,1,:,1] = ftree_array[[2,1],1,:,1]
                    
                    vs_right = ftree_data(f2)[1]

                    if vs_right == (0, 2, 1.0)
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
                vs_left = ftree_data(f1)[1]
                if iszero(vs_left[1])
                    # Swap columns (domain)
                    a[f1,f2][:,1,:,1] = ftree_array[:,1,[2,1],1]

                    if vs_left == (0, 2, 1.0)
                        # Flip sign to the first row (0,2,1)_1
                        a[f1,f2][1,1,:,1] *= -1
                    end
                else
                    # Swap rows (codomain)
                    a[f1,f2][:,1,:,1] = ftree_array[[2,1],1,:,1]
                    
                    vs_right = ftree_data(f2)[1]

                    if vs_right == (0, -2, 1.0)
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

Base.keys(lo2V::LocalOps_SpinSymm) = ["I", "c1", "c2", "a1", "a2"]
