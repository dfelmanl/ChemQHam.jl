using ITensorChemistry

name(atom::Atom) = atom.name
coordinates(atom::Atom) = atom.coordinates

# parse data contained in `mol` to generate
# an input string encoding for HF in Fermi.jl
function xyz_string(molecule::Molecule)
  molstr = ""
  for a in 1:length(molecule)
    atomname = name(molecule[a])
    atomcoords = Float64.(coordinates(molecule[a]))
    molstr *= atomname
    for r in atomcoords
      molstr *= " " * string(r)
    end
    molstr *= "\n"
  end
  return molstr
end

function molecular_interaction_coefficients(molecule::Molecule; kwargs...)
  return molecular_interaction_coefficients(
    xyz_string(Molecule(molecule)); kwargs...
  )
end

function molecular_interaction_coefficients(mol_str::String; basis="sto-3g", charge=0, spin=0)
    pyscf = PythonCall.pyimport("pyscf")

    if spin != 0
    error("Only spin=0 systems are handled right now!")
    end

    mol_obj = pyscf.gto.M(; atom=mol_str, basis=basis, charge=charge, spin=spin, verbose=2)

    # Run HF
    mf = pyscf.scf.RHF(mol_obj)
    mf.chkfile = ".tmpfile_pyscf"
    mf.kernel()

    # Create shorthands for 1- and 2-body integrals in MO basis
    mo = pyconvert(Array, mf.mo_coeff)
    hcore_ao = pyconvert(Array, mf.get_hcore())

    n = size(mo, 1)
    h1e = mo' * hcore_ao * mo
    h2e = reshape(pyconvert(Array, mol_obj.ao2mo(mf.mo_coeff; aosym=1)), n, n, n, n)

    # Collect data from HF calculation to return
    h2e = 0.5 * permutedims(h2e, (3, 2, 1, 4))
    nuc_e = pyconvert(Number, mf.energy_nuc())

    # println(pyconvert(String, mf.chkfile))
    rm(pyconvert(String, mf.chkfile))

    return h1e, h2e, nuc_e
end