using ChemQHam

println("=== Timing Analysis ===")

# Step 1: Molecular data calculation
@time begin
    molecule = Molecule([("Li", 0.00, 0.00, 0.0000), ("H", 0.00, 0.00, 1.000)])
    h1e, h2e, nuc_e, hf_orb_occ_basis, hf_elec_occ, hf_energy = molecular_hf_data(molecule)
end
println("✓ Molecular HF data calculated")

# Step 2: MPO construction
@time mpo = chemical_mpo(h1e, h2e, nuc_e; merge_physical_idx=false, clockwise_incoming_indices=false)
println("✓ MPO constructed")

# Step 3: HF MPS initialization
@time begin
    mps_hf = init_mps_hf(hf_orb_occ_basis, hf_elec_occ)
end
println("✓ HF MPS initialized")

# Step 4: DMRG with HF initial guess
@time begin
    mps0_hf = dmrg(mps_hf, mpo)
end
println("✓ DMRG with HF initialization completed")

# Step 5: Random MPS initialization and DMRG
@time begin
    mps_r = init_mps_rand(6, 4, 0)
end
println("✓ Random MPS initialized")

@time begin
    mps0_r = dmrg(mps_r, mpo)
end
println("✓ DMRG with random initialization completed")

dot = dotMPS(mps0_hf, mps0_r)
println("Dot product of MPS HF and MPS Random solutions:", dot)

println("=== Timing Analysis Completed ===")