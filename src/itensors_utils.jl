using ITensorChemistry: Molecule, molecular_orbital_hamiltonian

function ITChem_opsum(molecule::Molecule; n_sites::Int=0, idx_factor_list=nothing, it_terms_range=nothing, ops_tol=1e-8, ops_factor=1.0)
    ham_info = molecular_orbital_hamiltonian(molecule; basis="sto-3g", atol=ops_tol, n_sites=n_sites)
    opsum = ham_info.hamiltonian
    if !isnothing(idx_factor_list)
        opsum = edit_opsum(copy(opsum), idx_factor_list) * ops_factor
    elseif !isnothing(it_terms_range)
        opsum = reduce(+, op for op in opsum[it_terms_range]) * ops_factor
    end
    # println("opsum: ", opsum)
    return (opsum=opsum, h1e=ham_info.h1e, h2e=ham_info.h2e, nuc_e=ham_info.nuclear_energy)
end

function ITChem_mat(opsum, n_sites; ham_tol=1e-10, ham_maxdim=10000)
    sites = siteinds("Electron", n_sites; conserve_qns=true)
    H_mpo = ITensorMPS.MPO(opsum, sites, cutoff=ham_tol, maxdim=ham_maxdim)
    H_tens = reduce(*, H_mpo);
    mpo_sites = vcat([dag(p_ind) for p_ind in sites],[p_ind' for p_ind in sites])
    H_sparse = sparse(reshape(Array(H_tens, mpo_sites), (4^n_sites,4^n_sites)))
    return H_sparse
end
