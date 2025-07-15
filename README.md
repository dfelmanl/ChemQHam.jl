# ChemQHam.jl

A Julia package for optimal quantum chemistry Hamiltonian construction in matrix product operator (MPO) form.

⚠️ **This package is currently under active development** ⚠️

## Features

- **Chemical Operator Sum Generation**: Convert quantum chemistry integrals to operator sums
- **Optimized MPO Construction**: Uses bipartite graph theory methods for minimal bond dimension
- **Minimal Hamiltonian terms**: Takes advantage of the symmetry in the two-electron interaction coefficients
- **TensorKit Integration**: Convert to TensorKit tensor network format with symmetry support
- **fZ⊗U₁⊗SU₂ Symmetry**: Currently supports U₁ particle number ⊗ SU₂ spin rotation symmetry on top of the fundamental fermionic parity
- **ITensor Compatibility**: Interface with ITensorMPS and ITensorChemistry for testing

## Installation

```bash
julia> using Pkg

julia> Pkg.add(; url="https://github.com/dfelmanl/ChemQHam.jl")
```

## Quick Start Example

```julia
using ChemQHam

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

- **Build the ops table according to the virtual indexes when having spin symmetry**: When having spin symmetry, the virtual indexes are not unique, as they can represent multiple spin sectors. If we take each unique virtual space path individually, we will be able to allow for larger terms grouping, which results in a lower _minimum vertex cover_ for the bipartite algorithm.

## Acknowledgments
- **Dr. Philipp Schmoll** — for invaluable discussions and design insights throughout the development of this package.

- **Prof. Yaron Oz** and the Tel Aviv University — for the trust and financial support during this project.

- This package uses ideas from the **bipartite graph theory** algorithm described in [arXiv:2006.02056](https://arxiv.org/pdf/2006.02056) and implemented in the [Renormalizer](https://shuaigroup.github.io/Renormalizer/) project.
