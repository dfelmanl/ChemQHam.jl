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

# Generates a SparseBlockTensorMap object with fZ ⊠ U(1) ⊠ SU(2) symmetry
mpo = chemical_mpo(molecule)
```

## Next major improvements:

- Place ITensors dependencies in a new testing environment.
- Test if the sum of individual terms matrices equal the matrix from all terms together.

## Acknowledgments
- **Dr. Philipp Schmoll** — for invaluable discussions and design insights throughout the development of this package.

- **Prof. Yaron Oz** and the Tel Aviv University — for the trust and financial support during this project.

- This package uses ideas from the **bipartite graph theory** algorithm described in [arXiv:2006.02056](https://arxiv.org/pdf/2006.02056) and implemented in the [Renormalizer](https://shuaigroup.github.io/Renormalizer/) project.
