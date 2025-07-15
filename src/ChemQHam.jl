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

# Create and include first a symmetry.jl file
# include("symmetry.jl")
include("tensorkit_utils.jl")
include("chemical_data.jl")
include("opsum.jl")
include("terms_to_table.jl")
include("bipartite_vertex_cover.jl")
include("construct_symbolic_mpo.jl")
include("itensors_utils.jl")


# Export types
export ChemOpSum, OpTerm, Op2Data, Molecule

# Symbolic MPO related exports 
export molecular_interaction_coefficients, gen_ChemOpSum, terms_to_table, construct_symbolic_mpo

# TensorKit related exports
export chemical_mpo, symbolic_to_tensorkit_mpo, genPhySpace, mpo_to_mat

# ITensors related exports
export ITChem_opsum, ITChem_mat

end