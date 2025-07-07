module ChemQHam

using TensorKit
using ITensorChemistry
using ITensorMPS
using ITensors
using HTTN # Only required for the SparseMPO function, which may not be necessary
using LinearAlgebra
using SparseArrays
using DataStructures: DefaultDict

include("opsum.jl")
include("terms_to_table.jl")
include("bipartite_vertex_cover.jl")
include("construct_symbolic_mpo.jl")
include("tensorkit_utils.jl")
include("itensors_utils.jl")


# Export types
export ChemOpSum, OpTerm, SiteOp
export ChemProperties

# Symbolic MPO related exports 
export gen_ChemOpSum, terms_to_table, construct_symbolic_mpo

# TensorKit related exports
export symbolic_to_tensorkit_mpo, genPhySpace, mpo_to_mat

# ITensors related exports
export ITChem_opsum, ITChem_mat

end