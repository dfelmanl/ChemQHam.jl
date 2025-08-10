using ITensorChemistry

name(atom::Atom) = atom.name
coordinates(atom::Atom) = atom.coordinates

"""
    xyz_string(molecule::Molecule)

Parse data contained in `molecule` to generate an input string encoding for HF calculation.

# Arguments
- `molecule::Molecule`: The molecule object containing atoms and their coordinates

# Returns
- `molstr::String`: A formatted string representation of the molecule for quantum chemistry calculations
"""
function xyz_string(molecule::Molecule)
    molstr = ""
    for a = 1:length(molecule)
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

"""
    molecular_hf_data(molecule::Molecule; kwargs...)

Compute Hartree-Fock data for a molecule.

# Arguments
- `molecule::Molecule`: The molecule object

# Returns
- Hartree-Fock calculation results including integrals and occupation data
"""
function molecular_hf_data(molecule::Molecule; kwargs...)
    return molecular_hf_data(xyz_string(Molecule(molecule)); kwargs...)
end

"""
    molecular_hf_data(mol_str::String; basis="sto-3g", charge=nothing, spin=nothing)

Compute Hartree-Fock data for a molecule from its string representation.

# Arguments
- `mol_str::String`: String representation of the molecule geometry
- `basis::String="sto-3g"`: Basis set to use for the calculation
- `charge::Union{Py, Int}=PythonCall.Py(nothing)`: Molecular charge
- `spin::Union{Py, Int}=PythonCall.Py(nothing)`: Molecular spin

# Returns
- `h1e`: One-electron integrals in molecular orbital basis
- `h2e`: Two-electron integrals in molecular orbital basis  
- `nuc_e`: Nuclear repulsion energy
- `hf_orb_occ_basis`: Hartree-Fock orbital occupation basis
- `hf_elec_occ`: Hartree-Fock electron occupation
- `hf_energy`: Hartree-Fock total energy
"""
function molecular_hf_data(
    mol_str::String;
    basis::String = "sto-3g",
    charge::Union{Py,Int} = PythonCall.Py(nothing),
    spin::Union{Py,Int} = PythonCall.Py(nothing),
)
    pyscf = PythonCall.pyimport("pyscf")

    mol_obj = pyscf.gto.M(;
        atom = mol_str,
        basis = basis,
        charge = charge,
        spin = spin,
        verbose = 2,
    )

    # Run HF
    mf = pyscf.scf.RHF(mol_obj)
    mf.chkfile = ".tmpfile_pyscf"
    mf.kernel()

    # Create shorthands for 1- and 2-body integrals in MO basis
    mo = pyconvert(Array, mf.mo_coeff)
    hcore_ao = pyconvert(Array, mf.get_hcore())

    n_orb = size(mo, 1)
    h1e = mo' * hcore_ao * mo
    h2e = reshape(
        pyconvert(Array, mol_obj.ao2mo(mf.mo_coeff; aosym = 1)),
        n_orb,
        n_orb,
        n_orb,
        n_orb,
    )

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
    get_HF_occ(N_spt::Int, N_el::Int; init_ord::Vector{Int}=collect(1:N_spt))

Generate Hartree-Fock occupation numbers for a given number of spatial orbitals and electrons.

# Arguments
- `N_spt::Int`: Number of spatial orbitals
- `N_el::Int`: Number of electrons
- `init_ord::Vector{Int}=collect(1:N_spt)`: Initial orbital ordering

# Returns
- `occOrbSpace::Vector{Int}`: Orbital space occupation numbers
- `occNel::Vector{Int}`: Electron occupation numbers for each orbital
"""
function get_HF_occ(N_spt::Int, N_el::Int; init_ord::Vector{Int} = collect(1:N_spt))

    hf_occ = [Fill_HF(init_ord[p], N_el) for p = 1:N_spt]

    occOrbSpace, occNel = collect(zip(hf_occ...))

    return occOrbSpace, occNel

end

"""
    Fill_HF(i, nel)

Return Hartree-Fock filling tuple (orbital physical space basis, electron count) 
for spatial orbital i based on the total electron number.

# Arguments
- `i::Int`: Spatial orbital index
- `nel::Int`: Total number of electrons

# Returns
- Tuple of (orbital_space_state, electron_count) for the given orbital. Where the orbital states are: 
  - 4: Fully occupied (2 electrons, one spin-up and one spin-down)
  - 2: Half-occupied (1 electron, chosen to be spin-up). 3 would be spin-down, but the decision is arbitrary and does not affect the computation.
  - 1: Unoccupied (0 electrons)
"""
function Fill_HF(i, nel)
    if 2*i <= nel
        return 4, 2
    elseif 2*i-1 <= nel
        return 2, 1
    else
        return 1, 0
    end
end
