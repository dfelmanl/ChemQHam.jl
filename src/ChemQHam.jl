module ChemQHam

using ITensorChemistry
using ITensorMPS
using ITensors
using TensorKit
using BlockTensorKit
using LinearAlgebra
using SparseArrays
using DataStructures
using PythonCall
using Combinatorics

# Include files in dependency order
include("tensorkit_utils.jl")
include("symmetry.jl")
include("chemical_data.jl")
include("opsum.jl")
include("terms_to_table.jl")
include("bipartite_vertex_cover.jl")
include("construct_symbolic_mpo.jl")
include("chemical_mpo.jl")
include("itensors_utils.jl")


# Export types
export ChemOpSum, OpTerm, Op2Data, Molecule
export AbstractSymmetryContext, U1SU2SymmetryContext, U1U1SymmetryContext

# Symbolic MPO related exports 
export create_symmetry_context, get_operator_data, get_local_ops_idx_map, get_vs_idx_map
export get_all_local_ops, get_symmetry_name
export molecular_interaction_coefficients, gen_ChemOpSum, terms_to_table, construct_symbolic_mpo

# The main function
export chemical_mpo

# TensorKit related exports
export symbolic_to_tensorkit_mpo, genPhySpace, mpo_to_mat

# ITensors related exports
export ITChem_opsum, ITChem_mat

end