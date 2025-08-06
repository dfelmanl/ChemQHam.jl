# This is a temporary file for a DMRG implementation that uses BlockTensorMap and SparseBlockTensorMap.
# It is based on a private repository by Dr. Philipp Schmoll (https://github.com/philihps) and it was adapted for the current project.

# using HTTN
using KrylovKit
using Printf
using MPSKit
using TensorKit
using BlockTensorKit
using BlockTensorKit: ⊕
using SparseArrays
using LinearAlgebra

@kwdef struct DMRGCONFIG
    bondDim::Int64 = 10
    truncErr::Float64 = 1e-8
    convTolE::Float64 = 1e-6
    eigsTol::Float64 = 1e-12
    maxIterations::Int64 = 10
    subspaceExpansion::Bool = true
    verbose::Bool = false
    maxSweeps::Int64 = 10
    ovlpPenalty::Float64 = 2.0
end

# function initDMRG(mps::Vector{SparseBlockTensorMap}, H_mpo::Vector{SparseBlockTensorMap}; dmrg_sweeps::Int=10, ovlp_opt::Bool=false,
# ovlp_weight::Float64=2.0, phis::Vector{SparseMPS}=SparseMPS[], mps_maxdim::Int=32, truncErr::Float64=1e-8, verbose::Bool=false)
function initDMRG(mps::Vector{BlockTensorMap}, H_mpo::Vector{SparseBlockTensorMap}; dmrg_sweeps::Int=10, ovlp_opt::Bool=false,
                ovlp_weight::Float64=2.0, phis::Vector{Vector{BlockTensorMap}}=Vector{BlockTensorMap}[], mps_maxdim::Int=32, truncErr::Float64=1e-8, verbose::Bool=false)
    """
    Initialize MPS with DMRG.
    """

    if ovlp_opt && length(phis) > 0
        mps, energy, er = find_excitedstate(mps, H_mpo, phis,
                        DMRGCONFIG(; bondDim = mps_maxdim,
                        truncErr = 0., # truncErr=0. means truncation is only done by maxdim
                        verbose = verbose,
                        maxSweeps = dmrg_sweeps,
                        ovlpPenalty = ovlp_weight
                        ))
    else
        # Take DMRGCONFIG definition to the main script and apply initDMRG at a higher level, as it doesn't depend on hf_init
        mps, gsE = find_groundstate_wMaxdim(mps, H_mpo,
                            DMRGCONFIG(; bondDim = mps_maxdim,
                            truncErr = truncErr, # truncErr=0. means truncation is only done by maxdim
                            verbose = verbose,
                            maxSweeps = dmrg_sweeps
                            ))
    end

    return mps
end


# Redefine HTTN's find_groundstate to ignore truncation by tolerance (and fix bond dimensions)

# function find_groundstate_wMaxdim(finiteMPS::Vector{SparseBlockTensorMap}, finiteMPO::Vector{SparseBlockTensorMap}, alg::DMRGCONFIG)
function find_groundstate_wMaxdim(finiteMPS::Vector{BlockTensorMap}, finiteMPO::Vector{SparseBlockTensorMap}, alg::DMRGCONFIG; dataType::DataType=Float64)
    # dataType = eltype(finiteMPS[1].data)
    # @assert dataType == eltype(finiteMPO[1].data) "Unmatched: data from mps: $(typeof(finiteMPS[1].data)) and mpo: $(typeof(finiteMPO[1].data))"
    return find_groundstate_wMaxdim!(copy(finiteMPS), finiteMPO, alg; dataType=dataType)
end

# function find_groundstate_wMaxdim!(finiteMPS::Vector{SparseBlockTensorMap}, finiteMPO::Vector{SparseBlockTensorMap}, alg::DMRGCONFIG; dataType::DataType=Float64)
function find_groundstate_wMaxdim!(finiteMPS::Vector{BlockTensorMap}, finiteMPO::Vector{SparseBlockTensorMap}, alg::DMRGCONFIG; dataType::DataType=Float64)

    truncMethod = alg.truncErr > 0. ? truncdim(alg.bondDim) & truncerr(alg.truncErr) : truncdim(alg.bondDim)

    # apply finiteMPO to finiteMPS to introduce QNs that cannot be introduced by a regular 2-site update due to different local Hilbert spaces
    if alg.subspaceExpansion
        finiteMPS = applyMPO(finiteMPO, finiteMPS; maxDim=alg.bondDim, truncErr=alg.truncErr, # TO IMPLEMENT: split maxDim and truncErr between applyMPO and find_groundstate
                             compressionAlg = "zipUp")
    end
    # alg.verbose && println("MPS Spaces:\n $(space(finiteMPS[1]))\n $(space(finiteMPS[2]))\n $(space(finiteMPS[3]))\n $(space(finiteMPS[4]))\n $(space(finiteMPS[5]))\n $(space(finiteMPS[6]))")

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0]

    # store vector of truncation errors
    ϵs = ones(Float64, length(finiteMPS) - 1)

    # run DMRG until the energy variance is sufficiently close to 0
    optimizationLoopCounter = 1
    maxOptimSteps = 1
    runOptimization = true
    while runOptimization

        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO; dataType=dataType)

        # store vector of truncation errors
        ϵs = ones(Float64, length(finiteMPS) - 1)

        # main DMRG loop
        loopCounter = 1
        runOptimizationDMRG = true
        while runOptimizationDMRG

            # initialize vector to store energies
            newEigsEnergies = zeros(Float64, length(finiteMPS) - 1)

            # sweep L ---> R
            for siteIdx in 1:+1:(length(finiteMPS) - 1)

                # construct initial theta
                theta = TensorKit.permute(finiteMPS[siteIdx] *
                TensorKit.permute(finiteMPS[siteIdx + 1], (1,), (2, 3)),
                                (1, 2),
                                (3, 4))

                # optimize wave function to get newAC
                # eigenVal, eigenVec = HTTN.eigsolve(theta,
                eigenVal, eigenVec = eigsolve(theta,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = alg.maxIterations)) do x
                    # return HTTN.applyH2(x,
                    return applyH2(x,
                                   mpoEnvL[siteIdx],
                                   finiteMPO[siteIdx],
                                   finiteMPO[siteIdx + 1],
                                   mpoEnvR[siteIdx + 1])
                end
                eigVal = eigenVal[1]
                newTheta = eigenVec[1]
                newEigsEnergies[siteIdx] = eigVal

                # alg.verbose && println("newTheta: ", newTheta)

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  (1, 2),
                                  (3, 4);
                                  trunc = truncMethod,
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = TensorKit.permute(U, (1, 2), (3,))
                V = TensorKit.permute(S * V, (1, 2), (3,))



                # compute error
                v = @tensor theta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                # finiteMPS[siteIdx + 0] = SparseBlockTensorMap(U)
                # finiteMPS[siteIdx + 1] = SparseBlockTensorMap(V)
                finiteMPS[siteIdx + 0] = BlockTensorMap(U)
                finiteMPS[siteIdx + 1] = BlockTensorMap(V)

                # update mpoEnvL
                # mpoEnvL[siteIdx + 1] = SparseBlockTensorMap(HTTN.update_MPOEnvL(mpoEnvL[siteIdx],
                mpoEnvL[siteIdx + 1] = SparseBlockTensorMap(update_MPOEnvL(mpoEnvL[siteIdx],
                                                      finiteMPS[siteIdx],
                                                      finiteMPO[siteIdx],
                                                      finiteMPS[siteIdx]))
            end

            # sweep L <--- R
            for siteIdx in (length(finiteMPS) - 1):-1:1

                # construct initial theta
                theta = TensorKit.permute(finiteMPS[siteIdx] *
                TensorKit.permute(finiteMPS[siteIdx + 1], (1,), (2, 3)),
                                (1, 2),
                                (3, 4))

                # optimize wave function to get newAC
                # eigenVal, eigenVec = HTTN.eigsolve(theta,
                eigenVal, eigenVec = eigsolve(theta,
                                              1,
                                              :SR,
                                              KrylovKit.Lanczos(; tol = alg.eigsTol,
                                                                maxiter = alg.maxIterations)) do x
                    # return HTTN.applyH2(x,
                    return applyH2(x,
                                   mpoEnvL[siteIdx],
                                   finiteMPO[siteIdx],
                                   finiteMPO[siteIdx + 1],
                                   mpoEnvR[siteIdx + 1])
                end
                eigVal = eigenVal[1]
                newTheta = eigenVec[1]
                newEigsEnergies[siteIdx] = eigVal

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta,
                                  (1, 2),
                                  (3, 4);
                                  trunc = truncMethod,
                                  alg = TensorKit.SVD(),)
                S /= norm(S)
                U = TensorKit.permute(U * S, (1, 2), (3,))
                V = TensorKit.permute(V, (1, 2), (3,))

                # compute error
                v = @tensor theta[1, 2, 3, 4] * conj(U[1, 2, 5]) * conj(V[5, 3, 4])
                # ϵs[siteIdx] = max(ϵs[siteIdx], abs(1 - abs(v)));
                ϵs[siteIdx] = abs(1 - abs(v))

                # assign updated tensors and update MPO environments
                # finiteMPS[siteIdx + 0] = SparseBlockTensorMap(U)
                # finiteMPS[siteIdx + 1] = SparseBlockTensorMap(V)
                finiteMPS[siteIdx + 0] = BlockTensorMap(U)
                finiteMPS[siteIdx + 1] = BlockTensorMap(V)

                # update mpoEnvR
                # mpoEnvR[siteIdx + 0] = SparseBlockTensorMap(HTTN.update_MPOEnvR(mpoEnvR[siteIdx + 1],
                mpoEnvR[siteIdx + 0] = SparseBlockTensorMap(update_MPOEnvR(mpoEnvR[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1],
                                                      finiteMPO[siteIdx + 1],
                                                      finiteMPS[siteIdx + 1]))
            end

            # compute MPO expectation value
            mpoExpVal = expectation_value_mpo(finiteMPS, finiteMPO)
            if abs(imag(mpoExpVal)) < 1e-12
                mpoExpVal = real(mpoExpVal)
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal)

            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end])
            # energyConvergenceA = norm(newEigsEnergies .- oldEigsEnergies) / length(finiteMPS);
            # display([energyConvergence energyConvergenceA maximum(ϵs)])
            alg.verbose > 0 &&
                @printf("DMRG step %d ; energy = %0.8f ; convergence = %0.8e\n",
                        loopCounter,
                        mpoExpVal,
                        energyConvergence)
            if energyConvergence < alg.convTolE || loopCounter >= alg.maxSweeps
                runOptimizationDMRG = false
            end

            # increase loopCounter and update oldEigsEnergies
            loopCounter += 1
            # oldEigsEnergies = copy(newEigsEnergies);

        end

        # compute energy variance ⟨(H - E)^2⟩
        energyVariance = variance_mpo(finiteMPS, finiteMPO; trunc_err = alg.truncErr)
        alg.verbose && @printf("Energy variance ⟨ψ|(H - E)^2|ψ⟩ = %0.4e\n", energyVariance)

        # re-randomize finiteMPS if non-eigenstate was found
        # if energyVariance > 1e-0
        #     if optimizationLoopCounter < maxOptimSteps
        #         @printf("\nre-randomizing MPS...\n")
        #         for idxMPS in eachindex(finiteMPS)
        #             finiteMPS[idxMPS] += 0.1 *
        #                                  randn(dataType, codomain(finiteMPS[idxMPS]),
        #                                        domain(finiteMPS[idxMPS]))
        #         end
        #         finiteMPS = normalizeMPS(finiteMPS)
        #         finiteMPS = applyMPO(finiteMPO, finiteMPS; maxDim=alg.bondDim, truncErr=alg.truncErr,
        #                              compressionAlg = "zipUp")
        #     else
        #         runOptimization = false
        #     end
        # else
        #     runOptimization = false
        # end
        runOptimization = false;

        # increase optimizationLoopCounter
        optimizationLoopCounter += 1
    end
    # @printf("\n")

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end]
    return finiteMPS, finalEnergy, ϵs
end

# function initializeMPOEnvironments(finiteMPS::Vector{SparseBlockTensorMap}, finiteMPO::Vector{SparseBlockTensorMap};
function initializeMPOEnvironments(finiteMPS::Vector{BlockTensorMap}, finiteMPO::Vector{SparseBlockTensorMap};
                                   centerPos::Int64 = 1, dataType::DataType=Float64)

    # get length of finiteMPS
    N = length(finiteMPS)

    # construct MPO environments
    mpoEnvL = Vector{SparseBlockTensorMap}(undef, N)
    mpoEnvR = Vector{SparseBlockTensorMap}(undef, N)

    # initialize end-points of mpoEnvL and mpoEnvR
    mpoEnvL[1] = SparseBlockTensorMap(ones(dataType, space(finiteMPS[1], 1),
                      space(finiteMPO[1], 1) ⊗ space(finiteMPS[1], 1)))
    mpoEnvR[N] = SparseBlockTensorMap(ones(dataType,
                      space(finiteMPS[N], 3)' ⊗ space(finiteMPO[N], 3)',
                      space(finiteMPS[N], 3)'))

    # compute mpoEnvL up to (centerPos - 1)
    for siteIdx in 1:+1:(centerPos - 1)
        mpoEnvL[siteIdx + 1] = SparseBlockTensorMap(update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx],
                                              finiteMPO[siteIdx], finiteMPS[siteIdx]))
    end

    # compute mpoEnvR up to (centerPos + 1)
    for siteIdx in N:-1:(centerPos + 1)
        mpoEnvR[siteIdx - 1] = SparseBlockTensorMap(update_MPOEnvR(mpoEnvR[siteIdx], finiteMPS[siteIdx],
                                              finiteMPO[siteIdx], finiteMPS[siteIdx]))
    end
    return mpoEnvL, mpoEnvR
end

function update_MPOEnvR(mpoEnvR, mpsTensorK, mpoTensor, mpsTensorB)
    @tensor newER[-1 -2; -3] := mpsTensorK[-1, 2, 1] *
                                mpoTensor[-2, 4, 3, 2] *
                                conj(mpsTensorB[-3, 4, 5]) *
                                mpoEnvR[1, 3, 5]
    return newER
end

# function variance_mpo(finiteMPS::Vector{SparseBlockTensorMap}, finiteMPO::Vector{SparseBlockTensorMap}; trunc_err = 1e-6)
function variance_mpo(finiteMPS::Vector{BlockTensorMap}, finiteMPO::Vector{SparseBlockTensorMap}; trunc_err = 1e-6)

    # copy finiteMPS
    myMPS = copy(finiteMPS)

    # compute H|ψ⟩
    finiteMPOMPS = applyMPO(finiteMPO, myMPS; truncErr = trunc_err, compressionAlg = "zipUp")


    # compute E = ⟨ψ|H|ψ⟩
    # mpsEnergy = HTTN.dotMPS(myMPS, finiteMPOMPS)
    mpsEnergy = dotMPS(myMPS, finiteMPOMPS)

    # compute |ϕ⟩ = H|ψ⟩ - E|ψ⟩
    phiMPS = finiteMPOMPS - mpsEnergy * myMPS

    # compute variance as ⟨ϕ|ϕ⟩
    # energyVariance = abs(real(HTTN.dotMPS(phiMPS, phiMPS)))
    energyVariance = abs(real(dotMPS(phiMPS, phiMPS)))
    return energyVariance
end

# function HTTN.dotMPS(mpsA::Vector{SparseBlockTensorMap}, mpsB::Vector{SparseBlockTensorMap})
# function HTTN.dotMPS(mpsA::Vector{T}, mpsB::Vector{T}) where T <: SparseBlockTensorMap
# function HTTN.dotMPS(mpsA::Vector{BlockTensorMap}, mpsB::Vector{BlockTensorMap})
function dotMPS(mpsA::Vector{BlockTensorMap}, mpsB::Vector{BlockTensorMap})
    """ Compute the inner product < mpsA | mpsB > """
    """ Compute the overlap < mpsA | mpsB > """

    # get length of MPSs
    N = length(mpsA)
    N != length(mpsB) &&
        throw(DimensionMismatch("lengths of MPS A ($N) and MPS B ($length(mpsB)) do not match"))

    # initialize overlap
    overlapMPS = ones(Float64, space(mpsA[1], 1), space(mpsB[1], 1))
    for siteIdx in 1:N
        @tensor overlapMPS[-1; -2] := overlapMPS[1, 2] * conj(mpsA[siteIdx][1, 3, -1]) *
                                      mpsB[siteIdx][2, 3, -2]
    end
    overlapMPS = tr(overlapMPS)
    return overlapMPS
end


function applyMPO(finiteMPO::Vector{SparseBlockTensorMap},
                #   finiteMPS::Vector{SparseBlockTensorMap};
                  finiteMPS::Vector{BlockTensorMap};
                  truncErr::Float64 = 1e-6,
                  maxDim::Int64 = 2500,
                  compressionAlg::String = "variationalContraction",)
    """ Applies finiteMPO to finiteMPS and compressed the MPS to bond dimension 'maxDim' """
    """ Algorithm 'densityMatrix' is exact, algorithm 'zipUp' is faster but less accurate """

    # make copy of finiteMPS
    compressedMPS = copy(finiteMPS)

    # get length of finiteMPS
    N = length(compressedMPS)
    
    @assert compressionAlg == "zipUp"

    # construct left and right isomorphism
    isomoL = isomorphism(fuse(space(compressedMPS[1], 1), space(finiteMPO[1], 1)),
                            space(compressedMPS[1], 1) ⊗ space(finiteMPO[1], 1))
    isomoR = isomorphism(space(compressedMPS[N], 3)' ⊗ space(finiteMPO[N], 3)',
                            fuse(space(compressedMPS[N], 3)' ⊗ space(finiteMPO[N], 3)'))

    # zip-up from left to right
    for siteIdx in 1:N
        if siteIdx < N
            @tensor localTensor[-1 -2; -3 -4] := isomoL[-1, 1, 3] *
                                                    compressedMPS[siteIdx][1, 2, -3] *
                                                    finiteMPO[siteIdx][3, -2, -4, 2]
            U, S, V = tsvd(localTensor,
                            ((1, 2),
                            (3, 4));
                            trunc = truncdim(maxDim) & truncerr(truncErr), #to prevent an error for space mismatch
                            alg = TensorKit.SVD(),)
            # U, S, V = tsvd(localTensor, (1, 2), (3, 4), trunc = truncerr(truncErr), alg = TensorKit.SVD());
             compressed_mps= TensorKit.permute(U, ((1, 2), (3,)))
            #  compressedMPS[siteIdx] = SparseBlockTensorMap(compressed_mps)
             compressedMPS[siteIdx] = BlockTensorMap(compressed_mps)
            isomoL = S * V
        else
            @tensor compressed_mps[-1 -2; -3] := isomoL[-1, 1, 3] *
                                                            compressedMPS[siteIdx][1, 2,
                                                                                4] *
                                                            finiteMPO[siteIdx][3, -2, 5,
                                                                            2] *
                                                            isomoR[4, 5, -3]
            # compressedMPS[siteIdx] = SparseBlockTensorMap(compressed_mps)
            compressedMPS[siteIdx] = BlockTensorMap(compressed_mps)
        end
    end

    # orthogonalize MPS
    orthogonalizeMPS!(compressedMPS, 1)

    
    return compressedMPS
end

# function orthogonalizeMPS!(mps::Vector{SparseBlockTensorMap}, centerPos::Int64 = 1)
function orthogonalizeMPS!(mps::Vector{BlockTensorMap}, centerPos::Int64 = 1)
    for siteIdx in length(mps):-1:1
        (L, Q) = rightorth(mps[siteIdx], ((1,), (2, 3)); alg = LQpos())
        # normalizeMPS && normalize!(L)
        if siteIdx > 1
            # mps[siteIdx - 1] = SparseBlockTensorMap(TensorKit.permute(TensorKit.permute(mps[siteIdx - 1],
            mps[siteIdx - 1] = BlockTensorMap(TensorKit.permute(TensorKit.permute(mps[siteIdx - 1],
                                                ((1, 2),
                                                (3,))) * L, ((1, 2), (3,))))
            # mps[siteIdx - 0] = SparseBlockTensorMap(TensorKit.permute(Q, ((1, 2), (3,))))
            mps[siteIdx - 0] = BlockTensorMap(TensorKit.permute(Q, ((1, 2), (3,))))
        else
            # mps[siteIdx - 0] = SparseBlockTensorMap(TensorKit.permute(L * TensorKit.permute(Q, ((1,), (2, 3))),
            mps[siteIdx - 0] = BlockTensorMap(TensorKit.permute(L * TensorKit.permute(Q, ((1,), (2, 3))),
                                                ((1, 2),
                                                (3,))))
        end
    end
end


# addition
# function Base.:+(mpsA::Vector{SparseBlockTensorMap}, mpsB::Vector{SparseBlockTensorMap})
function Base.:+(mpsA::Vector{BlockTensorMap}, mpsB::Vector{BlockTensorMap})

    # get length of MPSs
    NA = length(mpsA)
    NB = length(mpsB)
    NA != NB &&
        throw(DimensionMismatch("lengths of MPS A ($NA) and MPS B ($NB) do not match"))

    # add MPSs from left to right
    # MPSC = Vector{SparseBlockTensorMap}(undef, NA)
    MPSC = Vector{BlockTensorMap}(undef, NA)
    if NA == 1

        # combine single tensor
        idxMPS = 1
        MPSC[idxMPS] = mpsA[idxMPS] + mpsB[idxMPS]

    else

        # left boundary tensor
        idxMPS = 1
        isoRA = isometry(space(mpsA[idxMPS], 3)' ⊕ space(mpsB[idxMPS], 3)',
                         space(mpsA[idxMPS], 3)')'
        isoRB = rightnull(isoRA)
        @tensor newTensor[-1 -2; -3] := mpsA[idxMPS][-1, -2, 3] * isoRA[3, -3] +
                                        mpsB[idxMPS][-1, -2, 3] * isoRB[3, -3]
        # MPSC[idxMPS] = SparseBlockTensorMap(newTensor)
        MPSC[idxMPS] = BlockTensorMap(newTensor)

        # bulk tensors
        for idxMPS in 2:(NA - 1)
            isoLA = isometry(space(mpsA[idxMPS], 1) ⊕ space(mpsB[idxMPS], 1),
                             space(mpsA[idxMPS], 1))
            isoLB = leftnull(isoLA)
            isoRA = isometry(space(mpsA[idxMPS], 3)' ⊕ space(mpsB[idxMPS], 3)',
                             space(mpsA[idxMPS], 3)')'
            isoRB = rightnull(isoRA)
            @tensor newTensor[-1 -2; -3] := isoLA[-1, 1] * mpsA[idxMPS][1, -2, 3] *
                                            isoRA[3, -3] +
                                            isoLB[-1, 1] * mpsB[idxMPS][1, -2, 3] *
                                            isoRB[3, -3]
            # MPSC[idxMPS] = SparseBlockTensorMap(newTensor)
            MPSC[idxMPS] = BlockTensorMap(newTensor)
        end

        # right boundary tensor
        idxMPS = NA
        isoLA = isometry(space(mpsA[idxMPS], 1) ⊕ space(mpsB[idxMPS], 1),
                         space(mpsA[idxMPS], 1))
        isoLB = leftnull(isoLA)
        @tensor newTensor[-1 -2; -3] := isoLA[-1, 1] * mpsA[idxMPS][1, -2, -3] +
                                        isoLB[-1, 1] * mpsB[idxMPS][1, -2, -3]
        # MPSC[idxMPS] = SparseBlockTensorMap(newTensor)
        MPSC[idxMPS] = BlockTensorMap(newTensor)
    end
    # return SparseMPS(MPSC)
    return MPSC
end
# Base.:-(mpsA::Vector{SparseBlockTensorMap}, mpsB::Vector{SparseBlockTensorMap}) = mpsA + (-1 * mpsB)
Base.:-(mpsA::Vector{BlockTensorMap}, mpsB::Vector{BlockTensorMap}) = mpsA + (-1 * mpsB)

# function Base.:*(ψ::Vector{SparseBlockTensorMap}, b::Number)
function Base.:*(ψ::Vector{BlockTensorMap}, b::Number)
    newTensors = copy(ψ)
    newTensors[1] *= b
    return newTensors
end
# Base.:*(b::Number, ψ::Vector{SparseBlockTensorMap}) = ψ * b
Base.:*(b::Number, ψ::Vector{BlockTensorMap}) = ψ * b



function applyH2(X, EL, mpo1, mpo2, ER)
    @tensor X[-1 -2; -3 -4] := EL[-1, 2, 1] * X[1, 3, 5, 6] * mpo1[2, -2, 4, 3] *
                               mpo2[4, -3, 7, 5] * ER[6, 7, -4]
    return X
end


function update_MPSEnvR(mpsEnvR, mpsTensorK, mpsTensorB)
    @tensor newER[-1; -2] := mpsTensorK[-1, 2, 1] * conj(mpsTensorB[-2, 2, 3]) *
                             mpsEnvR[1, 3]
    return newER
end

function update_MPOEnvL(mpoEnvL, mpsTensorK, mpoTensor, mpsTensorB)
    @tensor newEL[-1; -2 -3] := mpoEnvL[1, 3, 5] *
                                mpsTensorK[5, 4, -3] *
                                mpoTensor[3, 2, -2, 4] *
                                conj(mpsTensorB[1, 2, -1])
    return newEL
end

function expectation_value_mpo(finiteMPS, finiteMPO)
    """ Computes expectation value for MPS and MPO """

    # contract from left to right
    boundaryL = ones(ComplexF64, space(finiteMPS[1], 1),
                     space(finiteMPO[1], 1) ⊗ space(finiteMPS[1], 1))
    for siteIdx in 1:+1:length(finiteMPS)
        @tensor boundaryL[-1; -2 -3] := boundaryL[1, 3, 5] *
                                        finiteMPS[siteIdx][5, 4, -3] *
                                        finiteMPO[siteIdx][3, 2, -2, 4] *
                                        conj(finiteMPS[siteIdx][1, 2, -1])
    end
    boundaryR = ones(ComplexF64,
                     space(finiteMPS[end], 3)' ⊗ space(finiteMPO[end], 3)',
                     space(finiteMPS[end], 3)')

    # contact to get expectation value
    expectationVal = @tensor boundaryL[1, 2, 3] * boundaryR[3, 2, 1]
    return expectationVal
end
