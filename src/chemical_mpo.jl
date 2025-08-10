# Define local constant for default symmetry
const DEFAULT_SYMM = "U1SU2"

"""
    chemical_mpo(op_terms::ChemOpSum, symm_ctx::AbstractSymmetryContext; kwargs...)

Create a chemical MPO from a ChemOpSum with a given symmetry context.

# Arguments
- `op_terms::ChemOpSum`: Chemical operator sum
- `symm_ctx::AbstractSymmetryContext`: Symmetry context to use
- `algo::String="Hungarian"`: Algorithm for symbolic MPO construction
- `backend::String="TensorKit"`: Backend to use for numerical MPO
- `merge_physical_idx::Bool=false`: Whether to merge physical indices from multiple with dim=1 into one with dim=4
- `clockwise_incoming_indices::Bool=false`: Direction of incoming indices. Outgoing indices are always anti-clockwise.
- `dataType::DataType=Float64`: Data type for numerical tensors
- `verbose::Bool=false`: Whether to print verbose output

# Returns
- Backend-specific numerical MPO
"""
function chemical_mpo(op_terms::ChemOpSum,
                     symm_ctx::AbstractSymmetryContext;
                     algo::String="Hungarian",
                     backend::String="TensorKit",
                     merge_physical_idx::Bool=false,
                     clockwise_incoming_indices::Bool=false,
                     dataType::DataType=Float64,
                     verbose::Bool=false)
    
    # Create symbolic MPO
    symbolic_mpo, virt_spaces = construct_symbolic_mpo(op_terms, symm_ctx; algo=algo, verbose=verbose)
    
    # Convert to numerical MPO based on backend
    if uppercase(backend) == "TENSORKIT"

        numerical_mpo = symbolic_to_tensorkit_mpo(
            symbolic_mpo, virt_spaces, symm_ctx;
            merge_physical_idx=merge_physical_idx,
            clockwise_incoming_indices=clockwise_incoming_indices,
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
function chemical_mpo(mol_str::String; kwargs...)
    h1e, h2e, nuc_e, hf_orb_occ_basis, hf_elec_occ, hf_energy = molecular_hf_data(mol_str)
    return chemical_mpo(h1e, h2e, nuc_e; kwargs...)
end
chemical_mpo(h1e::AbstractArray{Float64}, h2e::AbstractArray{Float64}, nuc_e::Float64; kwargs...) = chemical_mpo(h1e, h2e; nuc_e=nuc_e, kwargs...)
