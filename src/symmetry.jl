# Symmetry context system for ChemQHam
# This file defines the symmetry context that holds all symmetry-specific information

# Abstract base type for all symmetry contexts
abstract type AbstractSymmetryContext end

"""
    AbstractSymmetryContext

Abstract base type for symmetry contexts. All concrete symmetry contexts should inherit from this type.
Each symmetry context should contain:
- name: String identifier for the symmetry
- operators: The local operators for this symmetry
- virtual_space_mappings: Pre-computed virtual space mappings
- physical_space_mappings: Physical space mappings
- local_ops_idx_map: Pre-computed mapping from operator strings to indices
- vs_idx_map: Pre-computed mapping from virtual space quantum numbers to indices
"""

"""
    U1SU2SymmetryContext

Symmetry context for U1×SU2 symmetry, commonly used in quantum chemistry.
This context pre-computes and stores all necessary mappings and data structures
needed for symbolic MPO construction.
"""
struct U1SU2SymmetryContext <: AbstractSymmetryContext
    name::String
    is_spin_symm::Bool
    operators::AbstractLocalOps{Float64}
    operator_data::Dict{String, Dict{Tuple{Any, Int}, Dict{Tuple{Any, Int}, Vector{Any}}}}
    local_ops_idx_map::Dict{String, Int}
    vs_idx_map::Dict{Tuple{Tuple{Bool, Int, Rational{Int64}}, Int64}, Int64}
    all_local_ops::Vector{String}
    is_filled::Dict{String, Bool}
    
    function U1SU2SymmetryContext(; fill_data::Bool=true)
        name = "U1SU2"
        is_spin_symm = true
        
        # Initialize operators
        ops = GetCrAnLocalOpsU1SU2()
        ops = LocalOps_DoubleV(ops)
        
        # Get all local operator strings
        all_local_ops = get_all_local_ops_str(name)
        
        # Create local operators index mapping
        local_ops_idx_map = Dict(op => i for (i, op) in enumerate(all_local_ops))
        
        # Create virtual space index mapping
        vs_idx_map = _compute_vs_idx_map_U1SU2()
        
        # Initialize operator data dictionary
        QNType = get_qn_type(name)
        OpDataDict = Dict{String, Dict{Tuple{QNType, Int}, Dict{Tuple{QNType, Int}, Vector{FusionTreeDataType(QNType)}}}}
        
        # Fill the operator data dictionary if requested
        if fill_data
            operator_data = OpDataDict()
            for op_str in all_local_ops
                op_data = get_op_data(ops, op_str, QNType)
                operator_data[op_str] = op_data
            end
            is_filled = Dict(op => true for op in all_local_ops)
        else
            operator_data = OpDataDict()
            is_filled = Dict(op => false for op in all_local_ops)
        end
        
        return new(name, is_spin_symm, ops, operator_data, local_ops_idx_map, vs_idx_map, all_local_ops, is_filled)
    end
end

"""
    U1U1SymmetryContext

Symmetry context for U1×U1 symmetry (particle number and spin conservation).
This context pre-computes and stores all necessary mappings and data structures
needed for symbolic MPO construction.
"""
struct U1U1SymmetryContext <: AbstractSymmetryContext
    name::String
    is_spin_symm::Bool
    operators::AbstractLocalOps{Float64}
    operator_data::Dict{String, Dict{Tuple{Any, Int}, Dict{Tuple{Any, Int}, Vector{Any}}}}
    local_ops_idx_map::Dict{String, Int}
    vs_idx_map::Dict{Any, Int}  # Will be properly typed when U1U1 is implemented
    all_local_ops::Vector{String}
    is_filled::Dict{String, Bool}
    
    function U1U1SymmetryContext(; fill_data::Bool=true)
        name = "U1U1"
        is_spin_symm = false
        
        # Initialize operators
        ops = GetCrAnLocalOpsU1U1(spin_symm=false)
        
        # Get all local operator strings
        all_local_ops = get_all_local_ops_str(name)
        
        # Create local operators index mapping
        local_ops_idx_map = Dict(op => i for (i, op) in enumerate(all_local_ops))
        
        # Create virtual space index mapping (TODO: implement when U1U1 is ready)
        vs_idx_map = Dict{Any, Int}()  # Placeholder
        
        # Initialize operator data dictionary
        QNType = get_qn_type(name)
        OpDataDict = Dict{String, Dict{Tuple{QNType, Int}, Dict{Tuple{QNType, Int}, Vector{FusionTreeDataType(QNType)}}}}
        
        # Fill the operator data dictionary if requested
        if fill_data
            operator_data = OpDataDict()
            for op_str in all_local_ops
                op_data = get_op_data(ops, op_str, QNType)
                operator_data[op_str] = op_data
            end
            is_filled = Dict(op => true for op in all_local_ops)
        else
            operator_data = OpDataDict()
            is_filled = Dict(op => false for op in all_local_ops)
        end
        
        return new(name, is_spin_symm, ops, operator_data, local_ops_idx_map, vs_idx_map, all_local_ops, is_filled)
    end
end

# Helper function to compute virtual space index mapping for U1SU2
function _compute_vs_idx_map_U1SU2()
    vs_dict = get_full_virt_space("U1SU2", as_dict=true)
    vs_idx_map = Dict{Tuple{Tuple{Bool, Int, Rational{Int64}}, Int64}, Int64}()
    idx = 1
    for (qn, mult) in vs_dict
        for i in 1:mult
            vs_idx_map[(qn, i)] = idx
            idx += 1
        end
    end
    return vs_idx_map
end

function get_virt_space_map_U1SU2()
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

# Factory function to create appropriate symmetry context
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

# Accessor functions for symmetry context
"""
    get_operator_data(symm_ctx::AbstractSymmetryContext, op_str::String)

Get the operator data for a given operator string from the symmetry context.
"""
function get_operator_data(symm_ctx::AbstractSymmetryContext, op_str::String)
    if haskey(symm_ctx.operator_data, op_str)
        return symm_ctx.operator_data[op_str]
    else
        # Lazy loading if data wasn't pre-computed
        if !symm_ctx.is_filled[op_str]
            QNType = get_qn_type(symm_ctx.name)
            op_data = get_op_data(symm_ctx.operators, op_str, QNType)
            symm_ctx.operator_data[op_str] = op_data
            symm_ctx.is_filled[op_str] = true
        end
        return symm_ctx.operator_data[op_str]
    end
end

"""
    get_local_ops_idx_map(symm_ctx::AbstractSymmetryContext)

Get the pre-computed local operators index mapping from the symmetry context.
"""
get_local_ops_idx_map(symm_ctx::AbstractSymmetryContext) = symm_ctx.local_ops_idx_map

"""
    get_vs_idx_map(symm_ctx::AbstractSymmetryContext)

Get the pre-computed virtual space index mapping from the symmetry context.
"""
get_vs_idx_map(symm_ctx::AbstractSymmetryContext) = symm_ctx.vs_idx_map

"""
    get_all_local_ops(symm_ctx::AbstractSymmetryContext)

Get all local operator strings for this symmetry.
"""
get_all_local_ops(symm_ctx::AbstractSymmetryContext) = symm_ctx.all_local_ops

"""
    get_symmetry_name(symm_ctx::AbstractSymmetryContext)

Get the name of the symmetry.
"""
get_symmetry_name(symm_ctx::AbstractSymmetryContext) = symm_ctx.name

# Compatibility function - allows indexing symmetry context like Op2Data
"""
    Base.getindex(symm_ctx::AbstractSymmetryContext, input::Tuple)

Allow indexing the symmetry context like Op2Data for backward compatibility.
"""
function Base.getindex(symm_ctx::AbstractSymmetryContext, input::Tuple{String, QN, Int, QN, Int}) where QN
    op_str, vs_out, vs_out_mult, vs_in, vs_in_mult = input
    
    op_data = get_operator_data(symm_ctx, op_str)
    
    if haskey(op_data, (vs_out, vs_out_mult))
        if haskey(op_data[(vs_out, vs_out_mult)], (vs_in, vs_in_mult))
            return op_data[(vs_out, vs_out_mult)][(vs_in, vs_in_mult)]
        end
    end
    
    # Return empty if not found
    return Vector{FusionTreeDataType(QN)}()
end
