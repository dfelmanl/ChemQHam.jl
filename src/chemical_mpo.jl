# Main interface for creating chemical MPOs
# This file provides the high-level interface for creating chemical MPOs

# Define local constant for default symmetry
const DEFAULT_SYMM = "U1SU2"

"""
    chemical_mpo(op_terms::ChemOpSum; kwargs...)

Create a chemical MPO from a ChemOpSum.

# Arguments
- `op_terms::ChemOpSum`: Chemical operator sum
- `symm::String="$DEFAULT_SYMM"`: Symmetry to use
- `backend::String="TensorKit"`: Backend to use for numerical MPO
- `algo::String="Hungarian"`: Algorithm for symbolic MPO construction
- `dataType::DataType=Float64`: Data type for numerical tensors
- `verbose::Bool=false`: Whether to print verbose output

# Returns
- Backend-specific numerical MPO
"""
function chemical_mpo(op_terms::ChemOpSum,
                     symm_context::AbstractSymmetryContext;
                     backend::String="TensorKit",
                     algo::String="Hungarian",
                     dataType::DataType=Float64,
                     verbose::Bool=false)
    
    # Create symbolic MPO
    symbolic_mpo, virt_spaces = construct_symbolic_mpo(op_terms, symm_context; algo=algo, verbose=verbose)
    
    # Convert to numerical MPO based on backend
    if uppercase(backend) == "TENSORKIT"
        vsQN_idx_map = get_vs_idx_map(symm_context)
        symm_name = get_symmetry_name(symm_context)
        
        # For now, we still need to create Op2Data for compatibility with existing tensorkit functions
        # TODO: Eventually update symbolic_to_tensorkit_mpo to use symmetry context directly
        op2data = Op2Data(symm_name)
        
        numerical_mpo = symbolic_to_tensorkit_mpo(
            symbolic_mpo, virt_spaces, symm_name, vsQN_idx_map, op2data; 
            dataType=dataType, verbose=verbose
        )
    else
        throw(ArgumentError("Unsupported backend: $backend. Supported backends are: 'TensorKit'."))
    end
    
    return numerical_mpo
end

chemical_mpo(op_terms::ChemOpSum, symm::String; kwargs...) = chemical_mpo(op_terms, create_symmetry_context(symm); kwargs...)
function chemical_mpo(op_terms::ChemOpSum; kwargs...)
    symm = get(kwargs, :symm, DEFAULT_SYMM)
    return chemical_mpo(op_terms, create_symmetry_context(symm); kwargs...)
end

"""
    chemical_mpo(h1e::AbstractArray{Float64}, h2e::AbstractArray{Float64}; kwargs...)

Create a chemical MPO from one-electron and two-electron integrals.

# Arguments
- `h1e::AbstractArray{Float64}`: One-electron integrals
- `h2e::AbstractArray{Float64}`: Two-electron integrals
- `nuc_e::Float64=0.0`: Nuclear repulsion energy
- `kwargs...`: Additional keyword arguments passed to chemical_mpo(op_terms; kwargs...)

# Returns
- Backend-specific numerical MPO
"""
function chemical_mpo(h1e::AbstractArray{Float64}, h2e::AbstractArray{Float64}; 
                     nuc_e::Float64=0.0, 
                     kwargs...)

    symm = get(kwargs, :symm, DEFAULT_SYMM)
    symm_ctx = create_symmetry_context(symm)
    
    # Create operator sum from integrals
    op_terms = gen_ChemOpSum(h1e, h2e, nuc_e; spin_symm=symm_ctx.is_spin_symm)
    
    # Dispatch to main implementation
    return chemical_mpo(op_terms, symm_ctx; kwargs...)
end

# Additional convenience functions for different input types
chemical_mpo(molecule::Molecule; kwargs...) = chemical_mpo(xyz_string(Molecule(molecule)); kwargs...)
chemical_mpo(mol_str::String; kwargs...) = chemical_mpo(molecular_interaction_coefficients(mol_str)...; kwargs...)
chemical_mpo(h1e::AbstractArray{Float64}, h2e::AbstractArray{Float64}, nuc_e::Float64; kwargs...) = chemical_mpo(h1e, h2e; nuc_e=nuc_e, kwargs...)
