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

function molecular_hf_data(molecule::Molecule; kwargs...)
  return molecular_hf_data(
    xyz_string(Molecule(molecule)); kwargs...
  )
end

function molecular_hf_data(mol_str::String; basis::String="sto-3g", charge::Union{Py, Int}=PythonCall.Py(nothing), spin::Union{Py, Int}=PythonCall.Py(nothing))
    pyscf = PythonCall.pyimport("pyscf")

    mol_obj = pyscf.gto.M(; atom=mol_str, basis=basis, charge=charge, spin=spin, verbose=2)

    # Run HF
    mf = pyscf.scf.RHF(mol_obj)
    mf.chkfile = ".tmpfile_pyscf"
    mf.kernel()

    # Create shorthands for 1- and 2-body integrals in MO basis
    mo = pyconvert(Array, mf.mo_coeff)
    hcore_ao = pyconvert(Array, mf.get_hcore())

    n_orb = size(mo, 1)
    h1e = mo' * hcore_ao * mo
    h2e = reshape(pyconvert(Array, mol_obj.ao2mo(mf.mo_coeff; aosym=1)), n_orb, n_orb, n_orb, n_orb)

    # Collect data from HF calculation to return
    h2e = 0.5 * permutedims(h2e, (3, 2, 1, 4))
    nuc_e = pyconvert(Number, mf.energy_nuc())

    # println(pyconvert(String, mf.chkfile))
    rm(pyconvert(String, mf.chkfile))

    # Calculate Hartree-Fock occupation numbers and energy
    n_elec = sum(pyconvert(Vector{Int}, mol_obj.nelec))
    hf_orb_occ_basis, hf_elec_occ = get_HF_occ(n_orb, n_elec)
    hf_energy = pyconvert(Number, mf.e_tot)


    return h1e, h2e, nuc_e, hf_orb_occ_basis, hf_elec_occ, hf_energy
end

"""
    getHFocc(chem_data)

Generate Hartree-Fock occupation numbers for a given chemical data.

# Arguments
- `chem_data::ChemData`: A structure containing chemical data, including the number of spatial orbitals (`N_spt`) and the number of electrons (`N_el`).

# Returns
- `occOrbSpace::Vector{Int}`: A vector containing the first elements of the Hartree-Fock occupation numbers.
- `occNel::Vector{Int}`: A vector containing the second elements of the Hartree-Fock occupation numbers.
"""
function get_HF_occ(N_spt::Int, N_el::Int; init_ord::Vector{Int}=collect(1:N_spt))
    
    hf_occ = [Fill_HF(init_ord[p], N_el) for p=1:N_spt]

    occOrbSpace, occNel = collect(zip(hf_occ...))
    
    return occOrbSpace, occNel
    
end

# Return HF filling tuple (orbitalPhysicalSpace basis, electron count) of the spatial orbital i occupation dependent on the electron number.
function Fill_HF(i, nel)
    if 2*i <= nel
        return 4, 2
    elseif 2*i-1 <= nel
        return 2, 1
    else
        return 1, 0
    end
end