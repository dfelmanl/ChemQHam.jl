
function init_mps_hf(occHF_orbBasis, occHF_elCnt; symm::String="U1SU2", dataType::DataType=Float64)
    """
    Initialize a Hartree Fock state MPS, which is a product state (bond dimension 1).
    """
    phy_sumspace = genFlattenedPhySpace(symm)
    N_spt = length(occHF_orbBasis)
    # construct virtual vector spaces for the MPS
    virtSpaces = Vector{SumSpace}(undef, N_spt + 1)
    virtSpaces[1] = oneunit(phy_sumspace)
    for siteIdx in 2:N_spt+1
        cummulative_n_el = sum(occHF_elCnt[1:siteIdx-1])

        if uppercase(symm) == "U1U1"
            fParity = cummulative_n_el % 2 == 0 ? 0 : 1
            virtSpaces[siteIdx] = SumSpace(Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])]((fParity, cummulative_n_el) => 1))
        elseif uppercase(symm) == "U1SU2"
            fParity = cummulative_n_el % 2 == 0 ? 0 : 1
            spin = fParity == 0 ? 0 : 1//2 # TODO: Fix this. The spin might not be always binary. We should get this from the getHFocc function
            virtSpaces[siteIdx] = SumSpace(Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((fParity, cummulative_n_el, spin) => 1))
        else
            error("Symmetry type $symm not supported")
        end
    end
    
    # create new classical product state from mpsSample
    # initialTensors = Vector{SparseBlockTensorMap}(undef, N_spt)
    initialTensors = Vector{BlockTensorMap}(undef, N_spt)
    for siteIdx in 1:N_spt
        virtSpaceL = virtSpaces[siteIdx]
        virtSpaceR = virtSpaces[siteIdx + 1]

        block_space = virtSpaceL ⊗ phy_sumspace ← virtSpaceR
        btm = zeros(block_space)

        physical_offset = occHF_orbBasis[siteIdx] == 4 ? 3 : occHF_orbBasis[siteIdx]
        idxs = CartesianIndex(1, physical_offset, 1)
        
        ftree_tm = TensorMap([1.], space(btm[idxs]))
        btm.data[idxs] = ftree_tm
        
        # spbTM=SparseBlockTensorMap(btm)
        # BlockTensorKit.dropzeros!(spbTM)
        # initialTensors[siteIdx] = spbTM
        
        initialTensors[siteIdx] = btm

    end
    # return SparseMPS(initialTensors; normalizeMPS = true)
    return initialTensors
end

function init_mps_rand(N_spt::Int, N_el::Int, spin; symm::String="U1SU2", removeDegeneracy::Bool=true, dataType::DataType=Float64)
    # Think about better ways to cutoff virtual BD states: with randomization / including different states accross MPSs
    """
    Initialize a random MPS with all bond dimensions.
    """
    phy_sumspace = genFlattenedPhySpace(symm)
    boundarySpaceL = oneunit(phy_sumspace)

    if uppercase(symm) == "U1U1"
        boundarySpaceR = SumSpace(Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])]((N_el % 2, N_el, spin) => 1))
    elseif uppercase(symm) == "U1SU2"
        boundarySpaceR = SumSpace(Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((N_el % 2, N_el, spin) => 1))
    else
        error("Symmetry type $symm not supported")
    end

    virtSpaces = constructVirtSpaces(phy_sumspace, N_spt, boundarySpaceL, boundarySpaceR;
                                    removeDegeneracy = removeDegeneracy);

    # initialize random MPS
    initialTensors = Vector{BlockTensorMap}(undef, N_spt);
    for siteIdx in 1:N_spt
        initialTensors[siteIdx] = randn(dataType, virtSpaces[siteIdx] ⊗ phy_sumspace,
                                        virtSpaces[siteIdx + 1])
    end

    # return SparseMPS(initialTensors; normalizeMPS = true) it will get normalized by FiniteMPS
    return initialTensors
end


function constructVirtSpaces(in_virtSpaces::Vector{V}, N_spt::Int, qnL::V, qnR::V;
                            removeDegeneracy::Bool = true,
                            degenCutOff::Int64 = 1) where {V<:ElementarySpace}
    """ Constructs vector spaces for virtual bond indices of the MPS """
    # TOIMPROVE: include maxdim (removing virtual qn with degeneracy 1)

    # create virtual vector spaces L
    virtSpaces_L = Vector{V}()
    virtSpaces_L = vcat(virtSpaces_L, qnL)
    for ind in 1:N_spt
        spaceL = virtSpaces_L[end]
        spaceR = fuse(spaceL, in_virtSpaces[ind])
        if removeDegeneracy
            spaceR = removeDegeneracyQN(spaceR; degenCutOff = degenCutOff)
        end
        virtSpaces_L = vcat(virtSpaces_L, [spaceR])
    end

    # create virtual vector spaces R
    virtSpaces_R = Vector{V}()
    virtSpaces_R = vcat(qnR, virtSpaces_R)
    for ind in 1:N_spt
        spaceR = virtSpaces_R[1]
        spaceL = fuse(conj(flip(in_virtSpaces[ind])), spaceR)
        if removeDegeneracy
            spaceL = removeDegeneracyQN(spaceL; degenCutOff = degenCutOff)
        end
        virtSpaces_R = vcat([spaceL], virtSpaces_R)
    end

    # combine virtual vector spaces
    # virtSpaces = [infimum(virtSpaces_L[siteIdx], virtSpaces_R[siteIdx])
    #               for siteIdx in 1:(N_spt + 1)]

    # combine virtual vector spaces
    virtSpaces = Vector{V}(undef, N_spt + 1)
    for siteIdx in 1:(N_spt + 1)
        println("Site $siteIdx:")
        println("  virtSpaces_L type: ", typeof(virtSpaces_L[siteIdx]))
        println("  virtSpaces_R type: ", typeof(virtSpaces_R[siteIdx]))
        println("  virtSpaces_L: ", virtSpaces_L[siteIdx])
        println("  virtSpaces_R: ", virtSpaces_R[siteIdx])
        println("  infimum type: ", typeof(infimum(virtSpaces_L[siteIdx], virtSpaces_R[siteIdx])))
        println("  infimum: ", infimum(virtSpaces_L[siteIdx], virtSpaces_R[siteIdx]))
        virtSpaces[siteIdx] = infimum(virtSpaces_L[siteIdx], virtSpaces_R[siteIdx])
    end

    return virtSpaces
end

function constructVirtSpaces(phySpace::S, N_spt::Int, boundarySpaceL::S, boundarySpaceR::S; removeDegeneracy::Bool=false) where {S<:SumSpace}

    phySpace_gs = sumSpace_to_gradedSpace(phySpace)
    boundarySpaceL_gs = sumSpace_to_gradedSpace(boundarySpaceL)
    boundarySpaceR_gs = sumSpace_to_gradedSpace(boundarySpaceR)


    phySpaces = fill(phySpace_gs, N_spt)

    virt_spaces_gs = constructVirtSpaces(phySpaces, N_spt, boundarySpaceL_gs, boundarySpaceR_gs; removeDegeneracy=removeDegeneracy)

    virt_spaces = Vector{S}(undef, N_spt + 1)
    for i in 1:(N_spt + 1)
        virt_spaces[i] = SumSpace(virt_spaces_gs[i])
    end

    return virt_spaces
end

function removeDegeneracyQN(vecSpace; degenCutOff::Int64 = 1)
    """ Function to remove degeneracies of QNs """

    qnSectors = vecSpace.dims
    truncatedVectorSpace = typeof(vecSpace)([key => min(qnSectors[key], degenCutOff) for key in keys(qnSectors)])
    return truncatedVectorSpace
end

# function removeDegeneracyQN(vecSpace; degenCutOff::Int64 = 1)
#     """ Function to remove degeneracies of QNs """

#     truncatedVectorSpaces = typeof(vecSpace.spaces)(undef, length(vecSpace.spaces))
#     for (i, space) in enumerate(vecSpace.spaces)
#         qnSectors = space.dims
#         truncatedVectorSpace = typeof(space)([key => min(qnSectors[key], degenCutOff) for key in keys(qnSectors)])
#         truncatedVectorSpaces[i] = truncatedVectorSpace
#     end
#     return SumSpace(truncatedVectorSpaces)
# end

function sumSpace_to_gradedSpace(vs_ss::SumSpace)
    """
    Convert a SumSpace to a GradedSpace.
    """
    # This function assumes that the SumSpace is already truncated
    # and has no degeneracies.
    
    qn_mults = Dict{ProductSector, Int64}()
    for (i, space) in enumerate(vs_ss.spaces)
        qnSectors = space.dims
        for key in keys(qnSectors)
            qn_mults[key] = qnSectors[key]
        end
    end
    vs_gs=typeof(vs_ss[1])(qn_mults)

    return vs_gs
end