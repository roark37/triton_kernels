import triton
import triton.language as tl

@triton.jit
def _att_fwd_inner(
    o_i, l_i, m_i, q_i, 
    k_block_ptr_base, v_block_ptr_base, 
    qk_scale, idx_row, T, 
    stage: tl.constexpr, BF16: tl.constexpr,
    dtype: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr,
):
    if stage == 0:   # no mask is used
        low, high = 0, T
        # when using tl.make_block_ptr(), the padding_option can not be "float('-inf')"
        # if T is not divisible by Bc, and the last block of K pads zeros, then the padded
        # zeros might cause the m_i value to be different from the correct one
        # so, we need to deal with the last block independently.
        # to avoid this complexity, we ask that T must be divisible by Bc.
    elif stage == 1: # processing tiles below the diagonal
        low, high = 0, idx_row * Br
    elif stage == 2: # processing tiles on the diagonal
        low, high = idx_row * Br, (idx_row + 1) * Br

    for id_col in tl.range(low, high, Bc):
        id_col = tl.multiple_of(id_col, Bc) 

        # Create K/V block pointers for this iteration 
        # Advance by (id_col, 0) because ptrs are now 2D (T, d)
        k_block_ptr = k_block_ptr_base.advance((id_col, 0))
        v_block_ptr = v_block_ptr_base.advance((id_col, 0))
        
        # load k
        # k_j will be (Bc, d) naturally
        k_j = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
        
        # Transpose k_j to (d, Bc) for dot product
        k_jT = tl.trans(k_j)

        # computation
        # the value of 'input_precision' arg in 'tl.dot' must be set at compile time, not runtime
        # the solution here is use a tl.constexpr type variable BF16 to determine the condition
        # BF16 is a compile-time constant, so the triton emits two different kernels for this condition
        # if BF16: 
        #     s_ij = tl.dot(q_i, k_jT) * qk_scale  # (Br, Bc)
        # else: 
        #     s_ij = tl.dot(q_i, k_jT, input_precision="ieee") * qk_scale
        s_ij = tl.dot(q_i, k_jT, input_precision="ieee") * qk_scale

        if stage == 2:  # only add mask to tiles on the diagonal
            offsets_h = id_col + tl.arange(0, Bc)
            offsets_v = idx_row * Br + tl.arange(0, Br)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf')) 
        
        m_ij = tl.max(s_ij, axis=1) # (Br, )
        m_i_new = tl.maximum(m_i, m_ij) # (Br, )
        p_ij = tl.exp(s_ij - m_i_new[:, None])   # (Br, Bc)

        l_ij = tl.sum(p_ij, axis=1)       # (Br, )
        adjust_factor = tl.exp(m_i - m_i_new)
        l_i = adjust_factor * l_i + l_ij
        m_i = m_i_new

        # load v
        # v_j will be (Bc, d) naturally
        v_j = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')

        p_ij = p_ij.to(dtype)

        # if BF16: 
        #     acc = tl.dot(p_ij, v_j)
        # else: 
        #     acc = tl.dot(p_ij, v_j, input_precision="ieee") 
        acc = tl.dot(p_ij, v_j, input_precision="ieee") 

        o_i = adjust_factor[:, None] * o_i + acc    # (Br, BLOCK_SIZE_D)
        
        # Warning!!! Large o_i would cause problem in this line of code:
        # Ex: data is float32 type, T=2048, d=16, q,k are all ones and v are consecutive integers, 
        # this line produce wrong o_i. when id_col == 90*Bc, tl.device_print('o_i', o_i)
        # print out that each row of o_i is    [16947840. 16949288. 16950752. 16952216. ... 16969688.].
        # While the corrected result should be [16947840. 16949296. 16950752. 16952208. ... 16969680.].
        # if insert a `tl.device_print('acc', acc) ` before 'o_i = adjust_factor[:, None] * o_i + acc', 
        # the code would get correct answer. 
        # And the output is correct when TRITON_INTERPRET=1
        
        # Explaination: might relate to how triton compiler processes 'o_i=... + acc' line differently
        #               with or wo the print operation, when one of the operands hit the precision wall

        #               When id_col=90, the value of the resulted o_i would cross the precision wall.
        #               from then on, the problem comes out.

        #               Precision wall: 
        #               when x > 2^24=16,777,216, it might not be represented precisely by float32,
        #               unless they happened to be divisible by 8, 16, 32, etc. the bigger the number, 
        #               the bigger the divisible denominator needs to be.

        #               In our case, it is strange that even though the value of elements in o_i
        #               hit the precision wall, the corrct value of them are happen to be those can be 
        #               represented by float32 precisely because all of them are divisible by 16.

        #               What the print operation changes?
        #               If a device_print operation is inserted before the o_i=... line, the compiler 
        #               would use different instructions to process the o_i=... line. 
        #               According to profiling result from NCU,there would be:
        #               1. more FMUL and FADD, less FFMA, which means less fused operations for optimization and 
        #               2. more registers used to save the intermediate results, and even cause the reg spilling
                               
        # ChatGPT think: The generated instruction sequence violates FP32 semantics when no device_print is used.
                               
        # When data is bloat16 type, still hit precision wall, but pytorch also hit. This would let the code
        # pass the test because the referred answer is wrong too.

        # # this kind of advance method would got wrong result, do not know why, just do not use
        # k_block_ptr = k_block_ptr.advance((0, Bc, 0))
        # v_block_ptr = v_block_ptr.advance((0, Bc, 0))
    return o_i, l_i, m_i

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, 
    O_ptr, L_ptr, 
    stride_qb, stride_qq, stride_qd, # shared by q, k, v, o
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vv, stride_vd,
    stride_ob, stride_oo, stride_od,
    stride_lb, stride_lq,      # different from stride of q, k, v, o
    N_QUERIES, N_KEYS,
    scale, 
    BF16: tl.constexpr,
    is_causal: tl.constexpr,
    D: tl.constexpr, # next_power_of_2(d)
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr, 
):
    # tl.static_print('block_size_D', BLOCK_SIZE_D)
    # tl.static_print('Bc', Bc)
    tl.static_assert(D >= K_TILE_SIZE) 
    dtype = tl.bfloat16 if BF16 else tl.float32

    idx_row, idx_bh = tl.program_id(0), tl.program_id(1)

    # 1. Manually advance pointers to the current Batch/Head (BH)
    #    This allows us to treat the remaining dimensions as a 2D grid (T, d)
    Q_ptr = Q_ptr + idx_bh * stride_qb
    K_ptr = K_ptr + idx_bh * stride_kb
    V_ptr = V_ptr + idx_bh * stride_vb
    O_ptr = O_ptr + idx_bh * stride_ob
    L_ptr = L_ptr + idx_bh * stride_lb

    # 2. Create 2D Block Pointers (Rank 2)
    #    This satisfies the layout constraints for tl.dot
    q_block_ptr = tl.make_block_ptr(
        Q_ptr, 
        shape=(N_QUERIES, D), 
        strides=(stride_qq, stride_qd), 
        block_shape=(Q_TILE_SIZE, D), 
        offsets=(idx_row * Q_TILE_SIZE, 0), 
        order=(1, 0), 
    )
    o_block_ptr = tl.make_block_ptr(
        O_ptr, 
        shape=(N_QUERIES, D), 
        strides=(stride_oo, stride_od), 
        block_shape=(Q_TILE_SIZE, D), 
        offsets=(idx_row * Q_TILE_SIZE, 0), 
        order=(1, 0), 
    )

    k_block_ptr_base = tl.make_block_ptr(
        K_ptr, 
        shape=(N_KEYS, D), 
        strides=(stride_kk, stride_kd), 
        block_shape=(K_TILE_SIZE, D), 
        offsets=(0, 0), 
        order=(1, 0), 
    )
    v_block_ptr_base = tl.make_block_ptr(
        V_ptr, 
        shape=(N_KEYS, D), 
        strides=(stride_vv, stride_vd), 
        block_shape=(K_TILE_SIZE, D),
        offsets=(0, 0), 
        order=(1, 0), 
    )
    # why order=(1, 0)?
    # 1. order Describes Memory Contiguity
    # The order parameter asks Triton: "Which dimensions are stored next to each other in physical RAM?" 
    # It expects a tuple sorted from the fastest-changing dimension (stride = 1) to the slowest.
    # Shape: (T, d)
    # Dimension 0: T (The rows)
    # Dimension 1: d (The columns / head dimension)
    # In a standard "Row-Major" (contiguous) tensor, the data for a single row is stored together.
    # Moving +1 in Dim 1 (d) moves to the next memory address (Stride = 1).
    # Moving +1 in Dim 0 (T) jumps over d items in memory (Stride = d).

    # load row blocks of Q
    # q_i loads as (Br, d), no reshape needed
    q_i = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')

    # init Oj, mj, lj
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) 
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)        
    m_i = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32) + float('-inf')

    if not is_causal: 
        o_i, l_i, m_i = _att_fwd_inner(
            o_i, l_i, m_i, q_i, 
            k_block_ptr_base, v_block_ptr_base, 
            scale, idx_row, N_QUERIES,
            stage=0, 
            BF16=BF16,
            dtype=dtype,
            Br=Q_TILE_SIZE, Bc=K_TILE_SIZE, 
        )
    else: 
        o_i, l_i, m_i = _att_fwd_inner(
            o_i, l_i, m_i, q_i, 
            k_block_ptr_base, v_block_ptr_base, 
            scale, idx_row, N_QUERIES,
            stage=1, 
            dtype=dtype, BF16=BF16,
            Br=Q_TILE_SIZE, Bc=K_TILE_SIZE, 
        )
        o_i, l_i, m_i = _att_fwd_inner(
            o_i, l_i, m_i, q_i, 
            k_block_ptr_base, v_block_ptr_base, 
            scale, idx_row, N_QUERIES,
            stage=2, 
            dtype=dtype, BF16=BF16,
            Br=Q_TILE_SIZE, Bc=K_TILE_SIZE, 
        )
    
    # compute o_i
    o_i = o_i / l_i[:, None] # (Br, d)
    o_i = o_i.to(dtype)

    L_i = m_i + tl.log(l_i)    # (Br, ) 
    
    # store results
    tl.store(o_block_ptr, o_i, boundary_check=(0, 1))

    # Store L using standard pointer arithmetic (easier for 1D vectors)
    L_row_offsets = idx_row * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    L_ptrs = L_ptr + L_row_offsets * stride_lq
    # Mask to ensure we don't write out of bounds if T is not a multiple of Br
    tl.store(L_ptrs, L_i, mask=L_row_offsets < N_QUERIES)

@triton.jit
def flash_bwd_pre_kernel(
    o_ptr, do_ptr, D_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    T, d, 
    B_outerloop: tl.constexpr, B_innerloop: tl.constexpr,
):
    """
    compute D = sum(o * do, axis=-1)
    """
    idx_row, idx_bh = tl.program_id(0), tl.program_id(1)

    # Compute base pointer offset for the current batch-head
    o_ptr_bh = o_ptr + idx_bh * stride_BH
    do_ptr_bh = do_ptr + idx_bh * stride_BH
    D_ptr_bh = D_ptr + idx_bh * D_stride_BH

    # Use 2D block pointers (T, d) for O and dO
    o_block_ptr = tl.make_block_ptr(
        o_ptr_bh, 
        shape=(T, d), 
        block_shape=(B_outerloop, B_innerloop),
        strides=(stride_T, stride_d), 
        offsets=(idx_row * B_outerloop, 0), 
        order=(1, 0), # move along the last dim (d) first
    )
    do_block_ptr = tl.make_block_ptr(
        do_ptr_bh, 
        shape=(T, d), 
        block_shape=(B_outerloop, B_innerloop),
        strides=(stride_T, stride_d), 
        offsets=(idx_row *B_outerloop, 0), 
        order=(1, 0), # move along the last dim (d) first
    )
    
    # Use 1D block pointer for D (just T dimension)
    D_block_ptr = tl.make_block_ptr(
        D_ptr_bh, 
        shape=(T,), 
        block_shape=(B_outerloop,),
        strides=(D_stride_T,), 
        offsets=(idx_row * B_outerloop,), 
        order=(0,),
    )

    D_block = tl.zeros((B_outerloop,), dtype=tl.float32)
    
    for id_col_inner in tl.range(0, d, B_innerloop):
        id_col_inner = tl.multiple_of(id_col_inner, B_innerloop)

        # Advance block pointers along the d dimension
        o_block_inner = tl.advance(o_block_ptr, (0, id_col_inner))
        do_block_inner = tl.advance(do_block_ptr, (0, id_col_inner))
        
        # Load data
        o_j = tl.load(o_block_inner, boundary_check=(0, 1), padding_option='zero')  # (Br, Bc)
        do_j = tl.load(do_block_inner, boundary_check=(0, 1), padding_option='zero')  # (Br, Bc)

        # Convert to float32 before computation
        o_j_fp32 = o_j.to(tl.float32)
        do_j_fp32 = do_j.to(tl.float32)

        # Compute in full precision
        D_block += tl.sum(o_j_fp32 * do_j_fp32, axis=1)  # (Br,)

    tl.store(D_block_ptr, D_block, boundary_check=(0,))


@triton.jit
def _bwd_dkdv_inner(
            q_block_ptr, L_block_ptr, D_block_ptr, do_block_ptr,
            k_jT, v_j, dk_j, dv_j,
            T, idx_col, 
            scale: tl.constexpr,
            BF16: tl.constexpr, dtype: tl.constexpr,
            B_outerloop: tl.constexpr, B_innerloop: tl.constexpr,
            stage: tl.constexpr,
        ):
    if stage == 0:
        low, high = 0, T
    elif stage == 1:
        low, high = idx_col * B_outerloop, (idx_col + 1) * B_outerloop
    elif stage == 2:
        low, high = (idx_col + 1) * B_outerloop, T

    # for id_step in tl.range(low, high, 1):
    #     id_row_inner = id_step * B_innerloop
    for id_row_inner in tl.range(low, high, B_innerloop):
        # advancing the base ptr
        q_block_ptr_inner = tl.advance(q_block_ptr, (id_row_inner, 0))
        do_block_ptr_inner = tl.advance(do_block_ptr, (id_row_inner, 0))
        L_block_ptr_inner = tl.advance(L_block_ptr, (id_row_inner,))
        D_block_ptr_inner = tl.advance(D_block_ptr, (id_row_inner,))

        # load data
        q_i = tl.load(q_block_ptr_inner, boundary_check=(0, 1), padding_option='zero')  # (Br, BLOCK_SIZE_D)
        do_i = tl.load(do_block_ptr_inner, boundary_check=(0, 1), padding_option='zero')  # (Br, BLOCK_SIZE_D)

        L_i = tl.load(L_block_ptr_inner, boundary_check=(0,), padding_option='zero')  # (Br,)
        D_i = tl.load(D_block_ptr_inner, boundary_check=(0,), padding_option='zero')  # (Br,)

        # re-compute p_ij
        # if BF16: 
        #     s_ij = tl.dot(q_i, k_jT) * scale
        # else: 
        #     s_ij = tl.dot(q_i, k_jT, input_precision="ieee") * scale
        s_ij = tl.dot(q_i, k_jT, input_precision="ieee") * scale

        if stage == 1: # only add mask at the first block when causal is True 
            offsets_h = idx_col * B_outerloop + tl.arange(0, B_outerloop)
            offsets_v = id_row_inner + tl.arange(0, B_innerloop)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf'))

        p_ij = tl.exp(s_ij - L_i[:, None])  # (Br, Bc), float32
        
        dtyped_p_ij = p_ij.to(dtype)

        # if BF16: 
        #     dv_j += tl.dot(dtyped_p_ij.T, do_i)  # (Bc, BLOCK_SIZE_D), float32
        #     dp_ij = tl.dot(do_i, v_j.T)  

        # else: 
        #     dv_j += tl.dot(dtyped_p_ij.T, do_i, input_precision="ieee")  # (Bc, BLOCK_SIZE_D), float32
        #     dp_ij = tl.dot(do_i, v_j.T, input_precision="ieee")
        dv_j += tl.dot(dtyped_p_ij.T, do_i, input_precision="ieee")  # (Bc, BLOCK_SIZE_D), float32
        dp_ij = tl.dot(do_i, v_j.T, input_precision="ieee")

        ds_ij = p_ij * (dp_ij - D_i[:, None])  # (Br, Bc), float32
        ds_ij = ds_ij.to(dtype)

        # if BF16: 
        #     dk_j += tl.dot(ds_ij.T, q_i)  # (Bc, BLOCK_SIZE_D), float32
        # else: 
        #     dk_j += tl.dot(ds_ij.T, q_i, input_precision="ieee")  # (Bc, BLOCK_SIZE_D), float32
        dk_j += tl.dot(ds_ij.T, q_i, input_precision="ieee")  # (Bc, BLOCK_SIZE_D), float32

    return dk_j, dv_j


@triton.jit
def flash_bwd_dkdv(
    q_ptr, k_ptr, v_ptr, dk_ptr, dv_ptr, 
    do_ptr, D_ptr, L_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    T, d, BF16: tl.constexpr,
    scale, is_causal, 
    BLOCK_SIZE_D: tl.constexpr,
    B_outerloop: tl.constexpr, B_innerloop: tl.constexpr,
):
    # ---------- Phase 1: compute dk, dv

    # outer loop: move across columns
    idx_col, idx_bh = tl.program_id(0), tl.program_id(1)

    dtype = tl.bfloat16 if BF16 else tl.float32

    # Compute base pointer offsets for the current batch-head
    k_ptr_bh = k_ptr + idx_bh * stride_BH
    v_ptr_bh = v_ptr + idx_bh * stride_BH
    q_ptr_bh = q_ptr + idx_bh * stride_BH
    dk_ptr_bh = dk_ptr + idx_bh * stride_BH
    dv_ptr_bh = dv_ptr + idx_bh * stride_BH
    do_ptr_bh = do_ptr + idx_bh * stride_BH
    L_ptr_bh = L_ptr + idx_bh * D_stride_BH
    D_ptr_bh = D_ptr + idx_bh * D_stride_BH

    # Use 2D block pointers (T, d)
    k_block_ptr = tl.make_block_ptr(
        k_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_outerloop, BLOCK_SIZE_D), 
        offsets=(idx_col * B_outerloop, 0), 
        order=(1, 0), 
    )
    v_block_ptr = tl.make_block_ptr(
        v_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_outerloop, BLOCK_SIZE_D),
        offsets=(idx_col * B_outerloop, 0), 
        order=(1, 0), 
    )
    q_block_ptr = tl.make_block_ptr(
        q_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_innerloop, BLOCK_SIZE_D),
        offsets=(0, 0), 
        order=(1, 0), 
    )
    dk_block_ptr = tl.make_block_ptr(
        dk_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_outerloop, BLOCK_SIZE_D), 
        offsets=(idx_col * B_outerloop, 0), 
        order=(1, 0), 
    )
    dv_block_ptr = tl.make_block_ptr(
        dv_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_outerloop, BLOCK_SIZE_D),
        offsets=(idx_col * B_outerloop, 0), 
        order=(1, 0), 
    )
    do_block_ptr = tl.make_block_ptr(
        do_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_innerloop, BLOCK_SIZE_D),
        offsets=(0, 0), 
        order=(1, 0), 
    )
    # Use 1D block pointers for L and D (just T dimension)
    L_block_ptr = tl.make_block_ptr(
        L_ptr_bh, 
        shape=(T,), 
        strides=(D_stride_T,), 
        block_shape=(B_innerloop,),
        offsets=(0,), 
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr_bh, 
        shape=(T,), 
        strides=(D_stride_T,), 
        block_shape=(B_innerloop,), 
        offsets=(0,), 
        order=(0,),
    )

    # load k_j and v_j
    k_j = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (Bc, BLOCK_SIZE_D)
    v_j = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (Bc, BLOCK_SIZE_D)
    k_jT = k_j.T  # (BLOCK_SIZE_D, Bc)

    # temporary variables
    dv_j = tl.zeros((B_outerloop, BLOCK_SIZE_D), dtype=tl.float32)
    dk_j = tl.zeros((B_outerloop, BLOCK_SIZE_D), dtype=tl.float32)

    if not is_causal:
        dk_j, dv_j = _bwd_dkdv_inner(
            q_block_ptr, L_block_ptr, D_block_ptr, do_block_ptr,
            k_jT, v_j, dk_j, dv_j,
            T, idx_col, 
            scale,
            BF16, dtype,
            B_outerloop, B_innerloop,
            stage=0,
        )
    else:
        dk_j, dv_j = _bwd_dkdv_inner(
            q_block_ptr, L_block_ptr, D_block_ptr, do_block_ptr,
            k_jT, v_j, dk_j, dv_j,
            T, idx_col, 
            scale,
            BF16, dtype,
            B_outerloop, B_innerloop,
            stage=1,
        )
        dk_j, dv_j = _bwd_dkdv_inner(
            q_block_ptr, L_block_ptr, D_block_ptr, do_block_ptr,
            k_jT, v_j, dk_j, dv_j,
            T, idx_col, 
            scale,
            BF16, dtype,
            B_outerloop, B_innerloop,
            stage=2,
        )
                
    # store dk, dv
    dk_j = dk_j * scale
    dk_j = dk_j.to(dtype)
    dv_j = dv_j.to(dtype)

    tl.store(dk_block_ptr, dk_j, boundary_check=(0, 1))
    tl.store(dv_block_ptr, dv_j, boundary_check=(0, 1))

@triton.jit
def _bwd_dq_inner(
            k_block_ptr_dq, v_block_ptr_dq, 
            q_i, do_i, L_i, D_i, dq_i,
            T, idx_row, 
            scale: tl.constexpr,
            BF16: tl.constexpr, dtype: tl.constexpr,
            B_outerloop: tl.constexpr, B_innerloop: tl.constexpr,
            stage: tl.constexpr,
        ):
    if stage == 0:
        low, high = 0, T
    elif stage == 1: 
        low, high = 0, idx_row * B_outerloop
    elif stage == 2:
        low, high = idx_row * B_outerloop, (idx_row + 1) * B_outerloop

    # inner loop: across column blocks
    for id_col_inner in tl.range(low, high, B_innerloop): 
        id_col_inner = tl.multiple_of(id_col_inner, B_innerloop)

        # update ptr
        k_block_ptr_inner = tl.advance(k_block_ptr_dq, (id_col_inner, 0))
        v_block_ptr_inner = tl.advance(v_block_ptr_dq, (id_col_inner, 0))

        # load data for one iter: k, v
        k_j = tl.load(k_block_ptr_inner, boundary_check=(0, 1), padding_option='zero')  # (Bc, BLOCK_SIZE_D)
        v_j = tl.load(v_block_ptr_inner, boundary_check=(0, 1), padding_option='zero')  # (Bc, BLOCK_SIZE_D)

        # recompute p
        s_ij = tl.dot(q_i, k_j.T, input_precision="ieee")  * scale # (Br, Bc), float32

        # if BF16: 
        #     s_ij = tl.dot(q_i, k_j.T)  * scale # (Br, Bc), float32
        # else: 
        #     s_ij = tl.dot(q_i, k_j.T, input_precision="ieee")  * scale # (Br, Bc), float32
        
        # only add mask to tiles on the diagonal
        if stage == 2:
            offsets_h = id_col_inner + tl.arange(0, B_innerloop)
            offsets_v = idx_row * B_outerloop + tl.arange(0, B_outerloop)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf'))

        p_ij = tl.exp(s_ij - L_i[:, None])

        dp_ij = tl.dot(do_i, v_j.T, input_precision="ieee")

        # if BF16: 
        #     dp_ij = tl.dot(do_i, v_j.T)  # (Br, Bc)
        # else: 
        #     dp_ij = tl.dot(do_i, v_j.T, input_precision="ieee")  # (Br, Bc)
        
        ds_ij = p_ij * (dp_ij - D_i[:, None])  # (Br, Bc), float32

        dq_i += tl.dot(ds_ij.to(dtype), k_j, input_precision="ieee")  # (Br, BLOCK_SIZE_D)

        # if BF16: 
        #     dq_i += tl.dot(ds_ij.to(dtype), k_j)  # (Br, BLOCK_SIZE_D)
        # else: 
        #     dq_i += tl.dot(ds_ij, k_j, input_precision="ieee")
    
    return dq_i

@triton.jit
def flash_bwd_dq(
    q_ptr, k_ptr, v_ptr, dq_ptr,
    do_ptr, D_ptr, L_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    T, d, BF16: tl.constexpr,
    scale: tl.constexpr, is_causal: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    B_outerloop: tl.constexpr, B_innerloop: tl.constexpr,
):
    idx_bh, idx_row = tl.program_id(1), tl.program_id(0)

    dtype = tl.bfloat16 if BF16 else tl.float32

    # Compute base pointer offsets for the current batch-head
    k_ptr_bh = k_ptr + idx_bh * stride_BH
    v_ptr_bh = v_ptr + idx_bh * stride_BH
    q_ptr_bh = q_ptr + idx_bh * stride_BH
    dq_ptr_bh = dq_ptr + idx_bh * stride_BH
    do_ptr_bh = do_ptr + idx_bh * stride_BH
    L_ptr_bh = L_ptr + idx_bh * D_stride_BH
    D_ptr_bh = D_ptr + idx_bh * D_stride_BH

    # Use 2D block pointers (T, d)
    k_block_ptr_dq = tl.make_block_ptr(
        k_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_innerloop, BLOCK_SIZE_D), 
        offsets=(0, 0), 
        order=(1, 0), 
    )
    v_block_ptr_dq = tl.make_block_ptr(
        v_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_innerloop, BLOCK_SIZE_D),
        offsets=(0, 0), 
        order=(1, 0), 
    )
    q_block_ptr_dq = tl.make_block_ptr(
        q_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_outerloop, BLOCK_SIZE_D),
        offsets=(idx_row * B_outerloop, 0), 
        order=(1, 0), 
    )
    dq_block_ptr_dq = tl.make_block_ptr(
        dq_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_outerloop, BLOCK_SIZE_D),
        offsets=(idx_row * B_outerloop, 0), 
        order=(1, 0), 
    )
    do_block_ptr_dq = tl.make_block_ptr(
        do_ptr_bh, 
        shape=(T, d), 
        strides=(stride_T, stride_d), 
        block_shape=(B_outerloop, BLOCK_SIZE_D),
        offsets=(idx_row * B_outerloop, 0), 
        order=(1, 0), 
    )
    # Use 1D block pointers for L and D (just T dimension)
    L_block_ptr_dq = tl.make_block_ptr(
        L_ptr_bh, 
        shape=(T,), 
        strides=(D_stride_T,), 
        block_shape=(B_outerloop,), 
        offsets=(idx_row * B_outerloop,), 
        order=(0,),
    )
    D_block_ptr_dq = tl.make_block_ptr(
        D_ptr_bh, 
        shape=(T,), 
        strides=(D_stride_T,), 
        block_shape=(B_outerloop,), 
        offsets=(idx_row * B_outerloop,), 
        order=(0,),
    )    
                        
    # load data for outer loop: move across row blocks
    q_i = tl.load(q_block_ptr_dq, boundary_check=(0, 1), padding_option='zero')  # (Br, BLOCK_SIZE_D)
    do_i = tl.load(do_block_ptr_dq, boundary_check=(0, 1), padding_option='zero')  # (Br, BLOCK_SIZE_D)

    L_i = tl.load(L_block_ptr_dq, boundary_check=(0,), padding_option='zero')  # (Br,), float32
    D_i = tl.load(D_block_ptr_dq, boundary_check=(0,), padding_option='zero')  # (Br,), float32

    dq_i = tl.zeros((B_outerloop, BLOCK_SIZE_D), dtype=tl.float32)

    if not is_causal:        
        dq_i = _bwd_dq_inner(
                k_block_ptr_dq, v_block_ptr_dq, 
                q_i, do_i, L_i, D_i, dq_i,
                T, idx_row, scale, 
                BF16=BF16, dtype=dtype,
                B_outerloop=B_outerloop, B_innerloop=B_innerloop,
                stage=0,
            )
    else:
        dq_i = _bwd_dq_inner(
                k_block_ptr_dq, v_block_ptr_dq, 
                q_i, do_i, L_i, D_i, dq_i,
                T, idx_row, scale, 
                BF16=BF16, dtype=dtype,
                B_outerloop=B_outerloop, B_innerloop=B_innerloop,
                stage=1,
            )
        dq_i = _bwd_dq_inner(
                k_block_ptr_dq, v_block_ptr_dq, 
                q_i, do_i, L_i, D_i, dq_i,
                T, idx_row, scale, 
                BF16=BF16, dtype=dtype,
                B_outerloop=B_outerloop, B_innerloop=B_innerloop,
                stage=2,
            )
    dq_i = (dq_i * scale).to(dtype)
    tl.store(dq_block_ptr_dq, dq_i, boundary_check=(0, 1))