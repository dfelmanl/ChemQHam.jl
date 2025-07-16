using Revise
using Pkg
Pkg.activate("..")
Pkg.instantiate()
Pkg.resolve()

include("../src/ChemQHam.jl")
using .ChemQHam



molecule = Molecule([("Li", 0.00, 0.00, 0.0000), ("H", 0.00, 0.00, 1.000)])
mpo = chemical_mpo(molecule)
println("MPO constructed successfully.")
println(mpo)