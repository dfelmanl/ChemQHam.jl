using Revise
using Pkg
Pkg.activate("..")
Pkg.instantiate()
Pkg.resolve()

include("../src/ChemQHam.jl")
using .ChemQHam

using ITensorChemistry
using SparseArrays
using LinearAlgebra

# Get Chemical data
n_sites = 4
ops_tol = 1e-8
ops_factor = 1

molecule = Molecule([("Li", 0.00, 0.00, 0.0000), ("H", 0.00, 0.00, 1.000)])
(; opsum, h1e, h2e, nuc_e) = ITChem_opsum(molecule, n_sites=n_sites, idx_factor_list=nothing, it_terms_range=nothing, ops_tol=ops_tol, ops_factor=ops_factor)

terms = gen_ChemOpSum(h1e, h2e, nuc_e; tol=1e-8)

# Convert to TensorKit MPO
symm="U1SU2"
algo="Hungarian"
verbose=false
op2data = Op2Data(symm)

table, factors, localOps_idx_map, vsQN_idx_map = terms_to_table(terms, op2data)
symbolic_mpo, virt_spaces = construct_symbolic_mpo(table, factors, localOps_idx_map, vsQN_idx_map, op2data; algo=algo, verbose=false);
mpo = symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, "u1su2", vsQN_idx_map, op2data; verbose=false)

# Convert MPO to matrix
it_mat = ITChem_mat(opsum)
tk_mat = mpo_to_mat(mpo)

# Compare matrices
norm_diff = norm(tk_mat-it_mat)
println("error = ", norm_diff)