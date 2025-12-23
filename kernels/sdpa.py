import triton
import triton.language as tl

@triton.jit
def _att_fwd_inner(
    o_i, l_i, m_i, q_i, 
    k_block_ptr_base, v_block_ptr_base, 
    qk_scale, idx_row, T, 
    stage: tl.constexpr, BF16: tl.constexpr,
    dtype: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,  
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

        # Create K/V block pointers for this iteration by advancing the base pointer
        k_block_ptr = k_block_ptr_base.advance((0, id_col, 0))
        v_block_ptr = v_block_ptr_base.advance((0, id_col, 0))
        
        # load k
        k_j = tl.load(k_block_ptr, boundary_check=(0, 1, 2), padding_option='zero')
        k_jT = k_j.reshape((Bc, BLOCK_SIZE_D)).T

        # computation
        # the value of 'input_precision' arg in 'tl.dot' must be set at compile time, not runtime
        # the solution here is use a tl.constexpr type variable BF16 to determine the condition
        # BF16 is a compile-time constant, so the triton emits two different kernels for this condition
        if BF16: 
            s_ij = tl.dot(q_i, k_jT) * qk_scale  # (Br, Bc)
        else: 
            s_ij = tl.dot(q_i, k_jT, input_precision="ieee")   # (Br, Bc)
            s_ij = s_ij * qk_scale

        # tl.device_print('s_ij: ', s_ij.to(tl.float32))
        # after dot operation, the dtype of s_ij is float32 even if the dtype of q_i and k_jT is float16
        
        if stage == 2:  # only add mask to tiles on the diagonal
            offsets_h = id_col + tl.arange(0, Bc)
            offsets_v = idx_row * Br + tl.arange(0, Br)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf')) 
        
        m_ij = tl.max(s_ij, axis=1) # (Br, )
        m_i_new = tl.maximum(m_i, m_ij) # (Br, )
        p_ij = tl.exp(s_ij - m_i_new[:, None])   # (Br, Bc)

        # if idx_row == 104: tl.device_print('p_ij', p_ij)

        l_ij = tl.sum(p_ij, axis=1)       # (Br, )
        adjust_factor = tl.exp(m_i - m_i_new)
        l_i = adjust_factor * l_i + l_ij
        m_i = m_i_new

        # load v
        v_j = tl.load(v_block_ptr, boundary_check=(0, 1, 2), padding_option='zero')
        v_j = v_j.reshape((Bc, BLOCK_SIZE_D))

        p_ij = p_ij.to(dtype)

        if BF16: 
            acc = tl.dot(p_ij, v_j)
        else: 
            acc = tl.dot(p_ij, v_j, input_precision="ieee") # dtype of p_ij and v_j should be the same
                                                            # acc is float32 even if the operands are float16
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
def att_fwd_kernel(
    qk_scale, q_ptr, k_ptr, v_ptr, o_ptr, L_ptr, 
    stride_BH, stride_T, stride_d, # shared by q, k, v, o
    L_stride_BH, L_stride_T,       # different from stride of q, k, v, o
    BH, T, d, 
    BF16: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, # next_power_of_2(d)
    Br: tl.constexpr, Bc: tl.constexpr, 
):
    tl.static_print('block_size_D', BLOCK_SIZE_D)
    tl.static_print('Bc', Bc)
    tl.static_assert(BLOCK_SIZE_D >= Bc) # for better performance of triton GEMM ops
    dtype = tl.bfloat16 if BF16 else tl.float32

    idx_row, idx_bh = tl.program_id(0), tl.program_id(1)

    q_block_ptr = tl.make_block_ptr(
        q_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d,), 
        # row block (Br, d)
        block_shape=(1, Br, BLOCK_SIZE_D), # block_shape中的元素得是constexpr
        offsets=(idx_bh, idx_row * Br, 0), 
        order=(1, 2, 0), # the iteration orde of dimensions
    )
    o_block_ptr = tl.make_block_ptr(
        o_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d), 
        # row block (Br, d)
        block_shape=(1, Br, BLOCK_SIZE_D),
        offsets=(idx_bh, idx_row * Br, 0), 
        order=(1, 2, 0), 
    )

    k_block_ptr_base = tl.make_block_ptr(
        k_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d,), 
        # row block (Bc, d)
        block_shape=(1, Bc, BLOCK_SIZE_D), 
        offsets=(idx_bh, 0, 0), 
        order=(1, 2, 0), 
    )
    v_block_ptr_base = tl.make_block_ptr(
        v_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d), 
        # row block (Bc, d)
        block_shape=(1, Bc, BLOCK_SIZE_D),
        offsets=(idx_bh, 0, 0), 
        order=(1, 2, 0), 
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr, 
        shape=(BH, T), 
        block_shape=(1, Br),
        strides=(L_stride_BH, L_stride_T), 
        offsets=(idx_bh, idx_row * Br), 
        order=(1, 0), 
    )

    # load row blocks of Q
    q_i = tl.load(q_block_ptr, boundary_check=(0, 1, 2), padding_option='zero')
    q_i = q_i.reshape((Br, BLOCK_SIZE_D)) # convert from 3D to 2D

    # init Oj, mj, lj
    o_i = tl.zeros((Br, BLOCK_SIZE_D), dtype=tl.float32) # use float32 for accumulation
    l_i = tl.zeros((Br, ), dtype=tl.float32)        
    m_i = tl.zeros((Br, ), dtype=tl.float32) + float('-inf')

    if not is_causal: 
        
        o_i, l_i, m_i = _att_fwd_inner(
            o_i, l_i, m_i, q_i, 
            k_block_ptr_base, v_block_ptr_base, 
            qk_scale, idx_row, T,
            stage=0, # flag for not causal
            BF16=BF16,
            dtype=dtype,
            Br=Br, Bc=Bc, BLOCK_SIZE_D=BLOCK_SIZE_D,  
        )
    else: 
        # processing the tiles below diag first then tiles on diag
        # Pass the base pointers (offset 0) to both stages. _att_fwd_inner calculates the correct offset.
        o_i, l_i, m_i = _att_fwd_inner(
            o_i, l_i, m_i, q_i, 
            k_block_ptr_base, v_block_ptr_base, 
            qk_scale, idx_row, T,
            stage=1, # processing tiles below diagonal
            dtype=dtype, BF16=BF16,
            Br=Br, Bc=Bc, BLOCK_SIZE_D=BLOCK_SIZE_D,  
        )
        o_i, l_i, m_i = _att_fwd_inner(
            o_i, l_i, m_i, q_i, 
            k_block_ptr_base, v_block_ptr_base, 
            qk_scale, idx_row, T,
            stage=2, # processing tiles on diagonal
            dtype=dtype, BF16=BF16,
            Br=Br, Bc=Bc, BLOCK_SIZE_D=BLOCK_SIZE_D,  
        )
    # compute o_i
    # if idx_row == 0: tl.device_print('o_i', o_i)
    o_i = o_i / l_i[:, None]# (Br, d)

    # if idx_row == 0: tl.device_print('o_i', o_i)

    o_i = o_i[None, :, :].to(dtype)

    L_i = m_i + tl.log(l_i)    # (Br, ) 
    # store results
    tl.store(o_block_ptr, o_i, boundary_check=(0, 1, 2))
    tl.store(L_block_ptr, L_i[None, :], boundary_check=(0, 1))

@triton.jit
def att_bwd_pre_kernel(
    o_ptr, do_ptr, D_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    BH, T, d, 
    Br: tl.constexpr, Bc: tl.constexpr,
):
    """
    compute D = sum(o * do, axis=-1)
    """
    idx_row, idx_bh = tl.program_id(0), tl.program_id(1)

    o_block_ptr = tl.make_block_ptr(
        o_ptr, 
        shape=(BH, T, d), 
        block_shape=(1, Br, Bc),
        strides=(stride_BH, stride_T, stride_d), 
        offsets=(idx_bh, idx_row * Br, 0), 
        order=(2, 1, 0), # move along the last dim
    )
    do_block_ptr = tl.make_block_ptr(
        do_ptr, 
        shape=(BH, T, d), 
        block_shape=(1, Br, Bc),
        strides=(stride_BH, stride_T, stride_d), 
        offsets=(idx_bh, idx_row * Br, 0), 
        order=(2, 1, 0), # move along the last dim
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr, 
        shape=(BH, T), 
        block_shape=(1, Br),
        strides=(D_stride_BH, D_stride_T), 
        offsets=(idx_bh, idx_row * Br), 
        order=(0, 1), # doesn't matter
    )

    D_block = tl.zeros((Br, ), dtype=tl.float32)
    for id_col_inner in tl.range(0, d, Bc):
        id_col_inner = tl.multiple_of(id_col_inner, Bc)

        o_block_inner = o_block_ptr.advance((0, 0, id_col_inner))
        do_block_inner = do_block_ptr.advance((0, 0, id_col_inner))
        # load data
        o_j = tl.load(o_block_inner, boundary_check=(0, 1, 2), padding_option='zero').reshape((Br, Bc))
        do_j = tl.load(do_block_inner, boundary_check=(0, 1, 2), padding_option='zero').reshape((Br, Bc))

        # Convert to float32 before computation
        o_j_fp32 = o_j.to(tl.float32)
        do_j_fp32 = do_j.to(tl.float32)

        # Now compute in full precision
        D_block += tl.sum(o_j_fp32 * do_j_fp32, axis=1) # (Br, )

    tl.store(D_block_ptr, D_block[None, :], boundary_check=(0, 1))

@triton.jit
def att_bwd_dkdv(
    q_ptr, k_ptr, v_ptr, dk_ptr, dv_ptr, 
    do_ptr, D_ptr, L_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    BH, T, d, BF16: tl.constexpr,
    scale, is_causal, 
    Multi_factor: tl.constexpr, 
    BLOCK_SIZE_D: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr,
):
    # ---------- Phase 1: compute dk, dv

    # outer loop: move across columns
    idx_row, idx_bh = tl.program_id(0), tl.program_id(1)

    dtype = tl.bfloat16 if BF16 else tl.float32

    k_block_ptr = tl.make_block_ptr(
        k_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d,), 
        # row block (Bc, d), move across rows in outer loop
        # use as column by transposing to k^T
        block_shape=(1, Bc, BLOCK_SIZE_D), 
        offsets=(idx_bh, idx_row * Bc, 0), 
        order=(1, 2, 0), 
    )
    v_block_ptr = tl.make_block_ptr(
        v_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d,), 
        # row block (Br, d), move across rows in outer loop
        # use as column by transposing to v^T
        block_shape=(1, Bc, BLOCK_SIZE_D),
        offsets=(idx_bh, idx_row * Bc, 0), 
        order=(1, 2, 0), 
    )
    q_block_ptr = tl.make_block_ptr(
        q_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d,), 
        # row block (Br, d), move across rows in innner loop
        block_shape=(1, Br, BLOCK_SIZE_D),
        offsets=(idx_bh, 0, 0), 
        order=(1, 2, 0), 
    )
    dk_block_ptr = tl.make_block_ptr(
        dk_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d,), 
        # row block (Bc, d), move across rows in outer loop
        block_shape=(1, Bc, BLOCK_SIZE_D), 
        offsets=(idx_bh, idx_row * Bc, 0), 
        order=(1, 2, 0), 
    )
    dv_block_ptr = tl.make_block_ptr(
        dv_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d), 
        # row block (Bc, d), move across rows in outer loop
        block_shape=(1, Bc, BLOCK_SIZE_D),
        offsets=(idx_bh, idx_row * Bc, 0), 
        order=(1, 2, 0), 
    )
    do_block_ptr = tl.make_block_ptr(
        do_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d), 
        # row block (Br, d), move across rows in innner loop
        block_shape=(1, Br, BLOCK_SIZE_D),
        offsets=(idx_bh, 0, 0), 
        order=(1, 2, 0), 
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr, 
        shape=(BH, T), 
        strides=(D_stride_BH, D_stride_T), 
        # load Br elements in each iter in the inner loop
        block_shape=(1, Br),
        offsets=(idx_bh, 0), 
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr, 
        shape=(BH, T), 
        strides=(D_stride_BH, D_stride_T), 
        # load Br elements in each iter in the inner loop
        block_shape=(1, Br), 
        offsets=(idx_bh, 0), 
        order=(1, 0),
    )

    # load k_j and v_j
    k_j = tl.load(k_block_ptr, boundary_check=(0, 1, 2), padding_option='zero')
    v_j = tl.load(v_block_ptr, boundary_check=(0, 1, 2), padding_option='zero')
    # convert to 2D
    k_jT = k_j.reshape((Bc, BLOCK_SIZE_D)).T # （BLOCK_SIZE_D, Bc）
    v_j  = v_j.reshape((Bc, BLOCK_SIZE_D))

    # temprary variables
    dv_j = tl.zeros((Bc, BLOCK_SIZE_D), dtype=tl.float32)
    dk_j = tl.zeros((Bc, BLOCK_SIZE_D), dtype=tl.float32)     # (BLOCK_SIZE_D, Bc)

    if is_causal:
        low, high = idx_row // Multi_factor, tl.cdiv(T, Br)
    else:
        low, high = 0, tl.cdiv(T, Br)

    for id_step in tl.range(low, high, 1):
        id_row_inner = id_step * Br
    
        # advancing the base ptr
        q_block_ptr_inner = q_block_ptr.advance((0, id_row_inner, 0))
        do_block_ptr_inner = do_block_ptr.advance((0, id_row_inner, 0))
        L_block_ptr_inner = L_block_ptr.advance((0, id_row_inner))
        D_block_ptr_inner = D_block_ptr.advance((0, id_row_inner))

        # load data
        q_i = tl.load(q_block_ptr_inner, boundary_check=(0, 1, 2), padding_option='zero')
        q_i = q_i.reshape((Br, BLOCK_SIZE_D))
        do_i = tl.load(do_block_ptr_inner, boundary_check=(0, 1, 2), padding_option='zero')
        do_i = do_i.reshape((Br, BLOCK_SIZE_D))

        L_i = tl.load(L_block_ptr_inner, boundary_check=(0, 1), padding_option='zero')
        D_i = tl.load(D_block_ptr_inner, boundary_check=(0, 1), padding_option='zero')
        L_i, D_i = L_i.reshape((Br, )), D_i.reshape((Br, ))

        # re-compute p_ij
        if BF16: 
            s_ij = tl.dot(q_i, k_jT)
        else: 
            s_ij = tl.dot(q_i, k_jT, input_precision="ieee") 

        if is_causal and (id_row_inner == low * Br): 
            offsets_h = idx_row * Bc + tl.arange(0, Bc)
            offsets_v = id_row_inner + tl.arange(0, Br)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf'))

        p_ij = tl.exp(s_ij - L_i[:, None])                  # (Br, Bc), float32
        
        dtyped_p_ij = p_ij.to(dtype)

        if BF16: 
            dv_j += tl.dot(dtyped_p_ij.T, do_i)  #  (Bc, BLOCK_SIZE_D), float32
            dp_ij = tl.dot(do_i, v_j.T)  

        else: 
            dv_j += tl.dot(dtyped_p_ij.T, do_i, input_precision="ieee") # (Bc, BLOCK_SIZE_D), float32
            dp_ij = tl.dot(do_i, v_j.T, input_precision="ieee")

        ds_ij = p_ij * (dp_ij - D_i[:, None]) # (Br, Bc), float32
        ds_ij = ds_ij.to(dtype)

        if BF16: 
            dk_j += tl.dot(ds_ij.T, q_i)  #  (Bc, BLOCK_SIZE_D), float32
        else: 
            dk_j += tl.dot(ds_ij.T, q_i, input_precision="ieee") # (Bc, BLOCK_SIZE_D), float32
                
    # store dk, dv
    dk_j = dk_j * scale
    dk_j = dk_j.to(dtype)
    dv_j = dv_j.to(dtype)

    tl.store(dk_block_ptr, dk_j[None, :, :], boundary_check=(0, 1, 2))
    tl.store(dv_block_ptr, dv_j[None, :, :], boundary_check=(0, 1, 2))


@triton.jit
def att_bwd_dq(
    q_ptr, k_ptr, v_ptr, dq_ptr,
    do_ptr, D_ptr, L_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    BH, T, d, BF16: tl.constexpr,
    scale, is_causal,
    BLOCK_SIZE_D: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr,
):
    idx_bh, idx_row = tl.program_id(1), tl.program_id(0)

    dtype = tl.bfloat16 if BF16 else tl.float32

    # re-create the ptrs for reversed outer and inner loop direction
    k_block_ptr_dq = tl.make_block_ptr(
        k_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d,), 
        # row block (Br, d), move across rows in inner loop
        # use as column by transposing to k^T
        block_shape=(1, Bc, BLOCK_SIZE_D), 
        offsets=(idx_bh, 0, 0), 
        order=(1, 2, 0), 
    )
    v_block_ptr_dq = tl.make_block_ptr(
        v_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d,), 
        # row block (Br, d), move across rows in inner loop
        # use as column by transposing to v^T
        block_shape=(1, Bc, BLOCK_SIZE_D),
        offsets=(idx_bh, 0, 0), 
        order=(1, 2, 0), 
    )
    q_block_ptr_dq = tl.make_block_ptr(
        q_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d), 
        # row block (Br, d), move across rows in outer loop
        block_shape=(1, Br, BLOCK_SIZE_D),  
        offsets=(idx_bh, idx_row * Br, 0), 
        order=(1, 2, 0), 
    )
    dq_block_ptr_dq = tl.make_block_ptr(
        dq_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d), 
        # row block (Br, d), move across rows in outer loop
        block_shape=(1, Br, BLOCK_SIZE_D),
        offsets=(idx_bh, idx_row * Br, 0), 
        order=(1, 2, 0), 
    )
    do_block_ptr_dq = tl.make_block_ptr(
        do_ptr, 
        shape=(BH, T, d), 
        strides=(stride_BH, stride_T, stride_d), 
        # row block (Br, d), move across rows in outer loop
        block_shape=(1, Br, BLOCK_SIZE_D),
        offsets=(idx_bh, idx_row * Br, 0), 
        order=(1, 2, 0), 
    )
    L_block_ptr_dq = tl.make_block_ptr(
        L_ptr, 
        shape=(BH, T), 
        strides=(D_stride_BH, D_stride_T), 
        # load Br elements in each iter in the outer loop
        block_shape=(1, Br),
        offsets=(idx_bh, idx_row * Br), 
        order=(1, 0),
    )
    D_block_ptr_dq = tl.make_block_ptr(
        D_ptr, 
        shape=(BH, T), 
        strides=(D_stride_BH, D_stride_T), 
        # load Br elements in each iter in the outer loop
        block_shape=(1, Br), 
        offsets=(idx_bh, idx_row * Br), 
        order=(1, 0),
    )    
                        
    # load data for outer loop: move acorss row blocks
    q_i = tl.load(q_block_ptr_dq, boundary_check=(0, 1, 2), padding_option='zero')
    do_i = tl.load(do_block_ptr_dq, boundary_check=(0, 1, 2), padding_option='zero')
    q_i, do_i = q_i.reshape((Br, BLOCK_SIZE_D)), do_i.reshape((Br, BLOCK_SIZE_D))

    L_i = tl.load(L_block_ptr_dq, boundary_check=(0, 1), padding_option='zero') # float32
    D_i = tl.load(D_block_ptr_dq, boundary_check=(0, 1), padding_option='zero') # float32
    L_i, D_i = L_i.reshape(Br, ), D_i.reshape(Br, )

    dq_i = tl.zeros((Br, BLOCK_SIZE_D), dtype=tl.float32) 
    if not is_causal:
        low, high = 0, T
        # if idx_row == 0: tl.device_print('no mask: high = ', T)
    else: 
        low, high = 0, (idx_row + 1) * Br
        # if idx_row == 0: tl.device_print('masked, high = ', high)

    # inner loop: across column blocks
    for id_col_inner in tl.range(low, high, Bc): 
        id_col_inner = tl.multiple_of(id_col_inner, Bc)

        # update ptr
        k_block_ptr_inner = k_block_ptr_dq.advance((0, id_col_inner, 0))
        v_block_ptr_inner = v_block_ptr_dq.advance((0, id_col_inner, 0))

        # load data for one iter: k, v
        k_j = tl.load(k_block_ptr_inner, boundary_check=(0, 1, 2), padding_option='zero')
        v_j = tl.load(v_block_ptr_inner, boundary_check=(0, 1, 2), padding_option='zero')
        # convert to 2D
        k_j, v_j = k_j.reshape((Bc, BLOCK_SIZE_D)), v_j.reshape((Bc, BLOCK_SIZE_D))
        # if idx_row == 0: tl.device_print('k_j = ', k_j)

        # recompute p
        if BF16: 
            s_ij = tl.dot(q_i, k_j.T)  # (Br, Bc)
        else: 
            s_ij = tl.dot(q_i, k_j.T, input_precision="ieee") # (Br, Bc)
        # mask the last (Br, Bc) tile
        if is_causal and id_col_inner == (high - Br):
            offsets_h = id_col_inner + tl.arange(0, Bc)
            offsets_v = idx_row * Br + tl.arange(0, Br)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf'))

        p_ij = tl.exp(s_ij - L_i[:, None])

        if BF16: 
            dp_ij = tl.dot(do_i, v_j.T)  # (Br, Bc)
        else: 
            dp_ij = tl.dot(do_i, v_j.T, input_precision="ieee") # (Br, Bc)
        
        ds_ij = p_ij * (dp_ij - D_i[:, None])                       # (Br, Bc), float32

        if BF16: 
            dq_i += tl.dot(ds_ij.to(dtype), k_j)  # (Br, BLOCK_SIZE_D)
        else: 
            dq_i += tl.dot(ds_ij.to(dtype), k_j, input_precision="ieee") # (Br, BLOCK_SIZE_D)
    
    dq_i = (dq_i * scale).to(dtype)

    tl.store(dq_block_ptr_dq, dq_i[None, :, :], boundary_check=(0, 1, 2))