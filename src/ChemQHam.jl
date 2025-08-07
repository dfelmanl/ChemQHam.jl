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
using KrylovKit
# using MPSKit
using BlockTensorKit: ⊕
using TensorKit: space

# Include files in dependency order
include("symmetry.jl")
include("chemical_data.jl")
include("opsum.jl")
include("terms_to_table.jl")
include("bipartite_vertex_cover.jl")
include("construct_symbolic_mpo.jl")
include("tensorkit_utils.jl")
include("chemical_mpo.jl")
include("itensors_utils.jl")
include("dmrg.jl")
include("init_mps.jl")


# Export types
export ChemOpSum, OpTerm, Molecule

# Symbolic MPO related exports 
export create_symmetry_context
export molecular_hf_data, gen_ChemOpSum, terms_to_table, construct_symbolic_mpo

# TensorKit related exports
export symbolic_to_tensorkit_mpo, mpo_to_mat

# The main function
export chemical_mpo

# DMRG related exports
export init_mps_hf, init_mps_rand, dmrg, dotMPS

# ITensors related exports
export ITChem_opsum, ITChem_mat

end