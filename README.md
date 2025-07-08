# ChemQHam.jl

A Julia package for optimal quantum chemistry Hamiltonian construction in matrix product operator (MPO) form.

⚠️ **This package is currently under active development** ⚠️

## Features

- **Chemical Operator Sum Generation**: Convert quantum chemistry integrals to operator sums
- **Optimized MPO Construction**: Uses bipartite graph theory methods for minimal bond dimension
- **TensorKit Integration**: Convert to TensorKit tensor network format with symmetry support
- **fZ⊗U₁⊗SU₂ Symmetry**: Currently supports U₁ particle number ⊗ SU₂ spin rotation symmetry on top of the fundamental fermionic parity
- **ITensor Compatibility**: Interface with ITensorMPS and ITensorChemistry for testing

## Quick Start Example

```julia
using Revise
using Pkg
Pkg.activate("..")
Pkg.instantiate()
Pkg.resolve()

include("../src/ChemQHam.jl")
using .ChemQHam

# Define the molecule
molecule = Molecule([("Li", 0.00, 0.00, 0.0000), ("H", 0.00, 0.00, 1.000)])

# Define the symmetry. Currently only U1SU2 is supported.
symm = "U1SU2"

# Generate molecular integrals
h1e, h2e, nuc_e = molecular_interaction_coefficients(molecule)

# Convert to chemical operator sum
op_terms = gen_ChemOpSum(h1e, h2e, nuc_e; tol=1e-8, spin_symm=true)

# Construct optimized symbolic MPO
symbolic_mpo, virt_spaces = construct_symbolic_mpo(op_terms)

# Convert to TensorKit MPO with symmetry
mpo = symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, symm)
```

## Next major improvements:

- **Use only the upper triangular hamiltonian coefficients**: Take advantage of the symmetry in the double-interaction hamiltonian coefficients. Given the two-electron integral $\nu_{pqrs}$, which corresponds to the term $\frac{1}{2} \nu_{pqrs} a_p^{\dagger} a_q^{\dagger}a_ra_s$, i.e. (ps|qr) in chemist's notation, there exists the symmetry: $g_{pqrs} = \nu_{pqrs} - \nu_{qprs} = \nu_{pqrs} - \nu_{pqsr}$. This reduces the double-interaction hamiltonian terms:
$$\hat{H}_{\text{2e}} = \frac{1}{2} \sum_{p,q,r,s=1}^{N} \nu_{pqrs} a_p^\dagger a_q^\dagger a_r a_s = \sum_{p<q, r<s}^{N} g_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$$
- **Build the tables according to the virtual indexes when having spin symmetry**: When having spin symmetry, the virtual indexes are not unique, as they can represent multiple spin sectors. If we take each unique virtual space path individually, we will be able to allow for larger terms grouping, which results in a lower _minimum vertex cover_ for the bipartite algorithm.