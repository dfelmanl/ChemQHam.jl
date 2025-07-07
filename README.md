# ChemQHam.jl

A Julia package for optimal quantum chemistry Hamiltonian construction in matrix product operator (MPO) form.

⚠️ **This package is currently under active development** ⚠️

## Features

- **Chemical Operator Sum Generation**: Convert quantum chemistry integrals to operator sums
- **Optimized MPO Construction**: Uses bipartite graph theory methods for minimal bond dimension
- **TensorKit Integration**: Convert to TensorKit tensor network format with symmetry support
- **fZ⊗U₁⊗SU₂ Symmetry**: Currently supports U₁ particle number ⊗ SU₂ spin rotation symmetry on top of the fundamental fermionic parity
- **ITensor Compatibility**: Interface with ITensorMPS and ITensorChemistry for testing
