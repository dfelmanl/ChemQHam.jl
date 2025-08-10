using Test
using ChemQHam

@testset "ChemQHam.jl Tests" begin
    @testset "Basic functionality" begin
        # Test basic molecule creation and MPO construction
        molecule = Molecule([("Li", 0.00, 0.00, 0.0000), ("H", 0.00, 0.00, 1.000)])

        # Test that we can get molecular data
        h1e, h2e, nuc_e, hf_orb_occ_basis, hf_elec_occ, hf_energy =
            molecular_hf_data(molecule)

        @test size(h1e, 1) > 0
        @test size(h2e, 1) > 0
        @test isa(nuc_e, Number)

        # Test MPO construction
        mpo = chemical_mpo(
            h1e,
            h2e,
            nuc_e;
            merge_physical_idx = false,
            clockwise_incoming_indices = false,
        )
        @test mpo !== nothing

        println("✓ Basic functionality tests passed")
    end

    @testset "MPS initialization" begin
        # Test MPS initialization functions using real molecular data
        molecule = Molecule([("Li", 0.00, 0.00, 0.0000), ("H", 0.00, 0.00, 1.000)])
        h1e, h2e, nuc_e, hf_orb_occ_basis, hf_elec_occ, hf_energy =
            molecular_hf_data(molecule)

        mps_hf = init_mps_hf(hf_orb_occ_basis, hf_elec_occ)
        @test mps_hf !== nothing

        # Test with realistic parameters for the random MPS
        N_spt = length(hf_orb_occ_basis)  # Number of spatial orbitals
        N_el = sum(hf_elec_occ)          # Total number of electrons
        mps_r = init_mps_rand(N_spt, N_el, 0)
        @test mps_r !== nothing

        println("✓ MPS initialization tests passed")
    end
end
