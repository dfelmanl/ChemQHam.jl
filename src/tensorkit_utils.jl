using BlockTensorKit: ⊕

function construct_empty_sparse_block_site(phySumSpace::SumSpace, left_vs::Vector{QN}, right_vs::Vector{QN}, symm::String; dataType::DataType=Float64) where QN
    """
    Construct an empty sparse block MPO site with the given quantum number multiplicities.
    
    Parameters:
    - phySpace: Physical space (GradedSpace) for TensorKit tensors
    - left_vs: individual left virtual spaces in an array
    - right_vs: individual right virtual spaces in an array
    - symm: Symmetry type, either "U1SU2" or "U1U1"
    
    Returns:
    - empty_block_tensor: An empty sparse block MPO site represented as a SparseBlockTensorMap
    """

    if symm == "U1SU2"
        # For U1SU2 symmetry, we use FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂]
        sum_Vspace_left = ⊕([Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])](vs => 1) for vs in first.(left_vs)]...)
        sum_Vspace_right = ⊕([Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])](vs => 1) for vs in first.(right_vs)]...)
    elseif symm == "U1U1"
        # For U1U1 symmetry, we use FermionParity ⊠ U1Irrep ⊠ U1Irrep
        sum_Vspace_left = ⊕([Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)](vs => 1) for vs in first.(left_vs)]...)
        sum_Vspace_right = ⊕([Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)](vs => 1) for vs in first.(right_vs)]...)
    else
        throw(ArgumentError("Unsupported symmetry type: $symm. Supported symmetries are 'U1SU2' and 'U1U1'."))
    end

    block_space = sum_Vspace_left ⊗ phySumSpace ← sum_Vspace_right ⊗ phySumSpace
    empty_block_tensor = BlockTensorKit.spzeros(dataType, block_space)

    return empty_block_tensor
end


function fill_mpo_site_SU2!(sblock_site, symb_mpo_site, vs_left, vs_right, symm_ctx::AbstractSymmetryContext, ftree_type, physical_abs_offsets; verbose=false)

    for (row, col, ops) in zip(findnz(symb_mpo_site)...)

        left_vsQN, left_vsMult = vs_left[row]
        right_qnQN, right_vsMult = vs_right[col]

        for (op_str, op_coef) in ops

            op_data = symm_ctx[(op_str, left_vsQN, left_vsMult, right_qnQN, right_vsMult)]

            for (ftree_left, ftree_right, val) in op_data

                ps_out = ftree_left[1][2] # physical sector out
                ps_right = ftree_right[1][2] # physical sector in

                phy_l = physical_abs_offsets[ps_out]
                phy_r = physical_abs_offsets[ps_right]

                sp_coordinates = (row, phy_l, col, phy_r)

                ftree_tm = copy(sblock_site[sp_coordinates...])
                old_val = only(ftree_tm[ftree_type(ftree_left...), ftree_type(ftree_right...)])
                new_val = old_val + val * op_coef
                ftree_tm[ftree_type(ftree_left...), ftree_type(ftree_right...)] .= new_val

                verbose && println("Filling MPO site for op_str: $op_str, left QN: $left_vsQN, left_op_mult: $left_vsMult, right QN: $right_qnQN, right_op_mult: $right_vsMult")
                verbose && println("sp_coordinates: $sp_coordinates, old_val: $old_val, new_val: $new_val, CG coef val: $val, operator's coef: $op_coef")
                push!(sblock_site.data, CartesianIndex(sp_coordinates...) => ftree_tm)
            end
        end
    end

end

function fill_mpo_site_U1U1!(mpo_site, symb_mpo_site, vs_map_left, vs_map_right, symm_ctx::AbstractSymmetryContext, ftree_type; verbose::Bool=false)

    for (row, col, ops) in zip(findnz(symb_mpo_site)...)
        (left_qn, left_mult) = vs_map_left[row]
        (right_qn, right_mult) = vs_map_right[col]
        
        for op in ops

            op_str = op.operator

            op_data = symm_ctx[(op_str, left_qn, right_qn)]

            if isempty(op_data)
                @warn "No data found for operator $(op.operator) with left QN $(left_qn) and right QN $(right_qn)."
                continue
            end
            for (ftree_left, ftree_right, val) in op_data
                f1 = ftree_type(ftree_left...)
                f2 = ftree_type(ftree_right...)

                verbose && println("Filling MPO site for operator: $op, op_str $op_str, left QN: $left_qn, right QN: $right_qn, left_mult: $left_mult, right_mult: $right_mult, ftree_left: $ftree_left, ftree_right: $ftree_right, value: $val, coef: $(op.coefficient)")
                mpo_site[f1,f2][left_mult,:,right_mult,:] .+= val * op.coefficient
                
            end
        end
    end

end

function symbolic_to_tensorkit_mpo(symbolic_mpo, virt_spaces, symm_ctx::AbstractSymmetryContext; merge_physical_idx::Bool=true, clockwise_incoming_indices::Bool=true, dataType::DataType=Float64, verbose=false)

    mpo_sites = Vector{SparseBlockTensorMap}(undef, length(symbolic_mpo)) # Allocate TensorMap with the already known properties (type, shape, etc.)

    phySumSpace = genFlattenedPhySpace(symm_ctx.name)
    physical_abs_offsets = get_physical_abs_offsets(symm_ctx.name)

    ftree_type = FusionTree{sectortype(phySumSpace)}

    for (isite, symb_mpo_site) in enumerate(symbolic_mpo)
        verbose && println("Processing MPO site $isite with symbolic data: $symb_mpo_site")
        vs_left = virt_spaces[isite]
        vs_right = virt_spaces[isite+1]
        
        sblock_site = construct_empty_sparse_block_site(phySumSpace, vs_left, vs_right, symm_ctx.name; dataType=dataType)
        
        if symm_ctx.name == "U1SU2"
            fill_mpo_site_SU2!(sblock_site, symb_mpo_site, vs_left, vs_right, symm_ctx, ftree_type, physical_abs_offsets; verbose=verbose)
        elseif symm_ctx.name == "U1U1"
            fill_mpo_site_U1U1!(sblock_site, symb_mpo_site, vs_map_left, vs_map_right, symm_ctx, ftree_type; verbose=verbose)
        else
            throw(ArgumentError("Unsupported symmetry type: $(symm_ctx.name)"))
        end
        
        mpo_sites[isite] = sblock_site
    end

    merge_physical_idx && merge_physical_idx!(mpo_sites, clockwise_codomain=false)

    clockwise_incoming_indices && flip_incoming_indices!(mpo_sites)

    return mpo_sites
end


function mpo_to_mat(H_mpo)
    N_spt = length(H_mpo)
    symm_space = TensorKit.space(H_mpo[1], 1)
    
    # Create boundary tensors with zero total quantum numbers
    boundaryMPO_L = TensorMap(ones, one(symm_space), oneunit(symm_space))
    boundaryMPO_R = TensorMap(ones, oneunit(symm_space), one(symm_space))
    
    H_contraction = boundaryMPO_L
    
    # Contract with remaining MPO tensors sequentially
    for i in 1:N_spt
        domain_legs = tuple(vcat(2*i+1, 1:(2*i-1))...)
        codomain_leg = (2*i,)
        
        H_contraction = TensorKit.permute(H_contraction * TensorKit.permute(H_mpo[i], (1,), (2,3,4)), 
                             domain_legs, codomain_leg)
    end
    
    right_legs = collect(N_spt+1:2*N_spt)
    left_legs = collect(N_spt:-1:1)
    
    H_contraction = TensorKit.permute(H_contraction * boundaryMPO_R, tuple(right_legs...), tuple(left_legs...))
    
    # Convert to TensorMap when working with `SparseBlockTensorMap`s
    if !isa(H_contraction, TensorMap)
        H_contraction = TensorMap(H_contraction)
    end

    # Reshape to matrix form
    mpo_mat = SparseArrays.sparse(reshape(convert(Array, H_contraction), (4^N_spt, 4^N_spt)))
    return mpo_mat
end

function genFlattenedPhySpace(symm)
    # This sets the convention for the order of each physical space sector.
    
    if uppercase(symm) == "U1"
        phySumSpace = Vect[(FermionParity ⊠ Irrep[U₁])]((0, 0) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁])]((1, 1) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁])]((1, 1) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁])]((0, 2) => 1)
    elseif uppercase(symm) == "U1U1"
        # (fParity, total count, spin count)
        phySumSpace = Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((0, 0, 0) => 1) ⊕ Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((1, 1, 1 // 2) => 1) ⊕ Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((1, 1, -1 // 2) => 1) ⊕ Vect[(FermionParity ⊠ U1Irrep ⊠ U1Irrep)]((0, 2, 0) => 1)
    elseif uppercase(symm) == "U1SU2"
        phySumSpace = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((1, 1, 1 // 2) => 1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 2, 0) => 1)
    else
        throw(ArgumentError("The specified symmetry type '$symm' is not implemented. The current options are: 'U1', 'U1U1', and 'U1SU2'"))
    end

    return phySumSpace

end

function get_physical_abs_offsets(symm::String)
    """
    Returns a function that maps the physical quantum numbers to their absolute offsets in the flattened physical space.
    """
    phySumSpace = genFlattenedPhySpace(symm)
    physical_offsets = Dict(ftree_inner_data(phySumSpace[i].dims.keys[1])  => i for i in 1:length(phySumSpace))
    
    return physical_offsets
end

function merge_site_physical_idx(mpo_site::SparseBlockTensorMap; clockwise_codomain::Bool=false)
    iso_dom = isometry(⊕(Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0)=>1, (1, 1, 1/2)=>1, (0, 2, 0)=>1)), ((Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0)=>1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((1, 1, 1/2)=>1) ⊕ Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 2, 0)=>1))))
    iso_codom = isometry(⊕(Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0)=>1, (1, 1, 1/2)=>1, (0, 2, 0)=>1)'), ((Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 0, 0)=>1)' ⊕ Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((1, 1, 1/2)=>1)' ⊕ Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂])]((0, 2, 0)=>1)')))
    if clockwise_codomain
        @tensor mpo_site_f_iso[-1,-2;-3,-4] := iso_dom[-2,1] * mpo_site[-1,1,2,-4] * iso_codom[-3,2]
    else
        @tensor mpo_site_f_iso[-1,-2;-3,-4] := iso_dom[-2,1] * mpo_site[-1,1,-3,2] * iso_codom[-4,2]
    end
    return SparseBlockTensorMap(mpo_site_f_iso)
end

function merge_physical_idx(mpo::Vector{<:SparseBlockTensorMap}; clockwise_codomain::Bool=false)
    merger(site::SparseBlockTensorMap) = merge_site_physical_idx(site; clockwise_codomain=clockwise_codomain)
    return merger.(mpo)
end

function merge_physical_idx!(mpo::Vector{<:SparseBlockTensorMap}; clockwise_codomain::Bool=false)
    for i in eachindex(mpo)
        mpo[i] = merge_site_physical_idx(mpo[i]; clockwise_codomain=clockwise_codomain)
    end
    return mpo
end

flip_incoming_indices(mpo_site::SparseBlockTensorMap) = TensorKit.permute(mpo_site, (1,2), (4,3))
flip_incoming_indices(mpo::Vector{<:SparseBlockTensorMap}) = flip_incoming_indices.(mpo)
function flip_incoming_indices!(mpo::Vector{<:SparseBlockTensorMap})
    for i in eachindex(mpo)
        mpo[i] = flip_incoming_indices(mpo[i])
    end
    return mpo
end