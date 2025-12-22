import triton
import triton.language as tl

@triton.jit
def _att_fwd_inner(
    o_i, l_i, m_i, q_i, 
    k_block_ptr_base, v_block_ptr_base, 
    qk_scale, idx_row, T, 
    stage,
    Br: tl.constexpr, Bc: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,  
):
    if stage == 0:   # no mask is used
        low, high = 0, T
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
        s_ij = tl.dot(q_i, k_jT, input_precision="ieee") * qk_scale  # (Br, Bc)
        
        if stage == 2:  # only need to add mask to tiles on the diagonal
            offsets_h = id_col + tl.arange(0, Bc)
            offsets_v = idx_row * Br + tl.arange(0, Br)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf')) 
        
        m_ij = tl.max(s_ij, axis=1) # (Br, )
        m_i_prev = m_i + 0.0
        m_i_new = tl.maximum(m_i_prev, m_ij) # (Br, )
        p_ij = tl.exp(s_ij - m_i_new[:, None])   # (Br, Bc)
        l_ij = tl.sum(p_ij, axis=1)       # (Br, )
        adjust_factor = tl.exp(m_i_prev - m_i_new)
        l_i = adjust_factor * l_i + l_ij
        m_i = m_i_new
        
        # load v
        v_j = tl.load(v_block_ptr, boundary_check=(0, 1, 2), padding_option='zero')
        v_j = v_j.reshape((Bc, BLOCK_SIZE_D))
        o_i = adjust_factor[:, None] * o_i + tl.dot(p_ij, v_j, input_precision="ieee")    # (Br, BLOCK_SIZE_D)

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
    is_causal: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, # next_power_of_2(d)
    Br: tl.constexpr, Bc: tl.constexpr, 
):
    tl.static_assert(BLOCK_SIZE_D >= Bc) # for better performance of triton GEMM ops
    idx_row = tl.program_id(0)
    idx_bh  = tl.program_id(1)

    # prepare block load vehicles
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
    # K/V pointers are initialized once at the base (offset 0)
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
    o_i = tl.zeros((Br, BLOCK_SIZE_D), dtype=tl.float32)
    l_i = tl.zeros((Br, ), dtype=tl.float32)
    m_i = tl.zeros((Br, ), dtype=tl.float32) + float('-inf')

    if not is_causal: 
        o_i, l_i, m_i = _att_fwd_inner(
            o_i, l_i, m_i, q_i, 
            k_block_ptr_base, v_block_ptr_base, 
            qk_scale, idx_row, T,
            stage=0, # flag for not causal
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
            Br=Br, Bc=Bc, BLOCK_SIZE_D=BLOCK_SIZE_D,  
        )
        o_i, l_i, m_i = _att_fwd_inner(
            o_i, l_i, m_i, q_i, 
            k_block_ptr_base, v_block_ptr_base, 
            qk_scale, idx_row, T,
            stage=2, # processing tiles on diagonal
            Br=Br, Bc=Bc, BLOCK_SIZE_D=BLOCK_SIZE_D,  
        )
    # compute o_i
    o_i = o_i / l_i[:, None]   # (Br, d)
    L_i = m_i + tl.log(l_i)    # (Br, ) 

    # store results
    tl.store(o_block_ptr, o_i[None, :, :], boundary_check=(0, 1, 2))
    tl.store(L_block_ptr, L_i[None, :], boundary_check=(0, 1))

@triton.jit
def att_bwd_pre_kernel(
    o_ptr, do_ptr, D_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    BH, T, d,
    BLOCK_SIZE_D: tl.constexpr, 
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

        # computation
        D_block += tl.sum(o_j * do_j, axis=1) # (Br, )

    tl.store(D_block_ptr, D_block[None, :], boundary_check=(0, 1))

@triton.jit
def att_bwd_dkdv(
    q_ptr, k_ptr, v_ptr, dk_ptr, dv_ptr, 
    do_ptr, D_ptr, L_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    BH, T, d,
    scale,
    Multi_factor: tl.constexpr, stage: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr,
):
    # ---------- Phase 1: compute dk, dv

    # outer loop: move across columns
    idx_bh, idx_row = tl.program_id(1), tl.program_id(0)
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

    if stage == 0:
        low, high = 0, tl.cdiv(T, Br)
    elif stage == 1:
        low, high = idx_row // Multi_factor, tl.cdiv(T, Br)

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
        s_ij = tl.dot(q_i, k_jT, input_precision="ieee")   # (Br, Bc)

        if stage == 1 and (id_row_inner == low * Br): 
            offsets_h = idx_row * Bc + tl.arange(0, Bc)
            offsets_v = id_row_inner + tl.arange(0, Br)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf'))

        p_ij = tl.exp(s_ij - L_i[:, None])                  # (Br, Bc)
        dv_j += tl.dot(p_ij.T, do_i, input_precision="ieee")# (Bc, BLOCK_SIZE_D), accumulating

        dp_ij = tl.dot(do_i, v_j.T, input_precision="ieee") # do_i@v_j^T, (Br, Bc)
        ds_ij = p_ij * (dp_ij - D_i[:, None])
        dk_j += tl.dot(ds_ij.T, q_i, input_precision="ieee")# (Bc, BLOCK_SIZE_D), accumulating
    
    # store dk, dv
    dk_j = dk_j * scale

    tl.store(dk_block_ptr, dk_j[None, :, :], boundary_check=(0, 1, 2))
    tl.store(dv_block_ptr, dv_j[None, :, :], boundary_check=(0, 1, 2))


@triton.jit
def att_bwd_dq(
    q_ptr, k_ptr, v_ptr, dq_ptr,
    do_ptr, D_ptr, L_ptr,
    stride_BH, stride_T, stride_d,
    D_stride_BH, D_stride_T,
    BH, T, d,
    scale, is_causal,
    BLOCK_SIZE_D: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr,
):
    idx_bh, idx_row = tl.program_id(1), tl.program_id(0)

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

    L_i = tl.load(L_block_ptr_dq, boundary_check=(0, 1), padding_option='zero')
    D_i = tl.load(D_block_ptr_dq, boundary_check=(0, 1), padding_option='zero')
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
        s_ij = tl.dot(q_i, k_j.T, input_precision='ieee') # (Br, Bc)
        # mask the last (Br, Bc) tile
        if is_causal and id_col_inner == (high - Br):
            offsets_h = id_col_inner + tl.arange(0, Bc)
            offsets_v = idx_row * Br + tl.arange(0, Br)
            mask = offsets_v[:, None] >= offsets_h[None, :]
            s_ij = s_ij + tl.where(mask, 0, float('-inf'))

        p_ij = tl.exp(s_ij - L_i[:, None])
        dp_ij = tl.dot(do_i, v_j.T, input_precision='ieee')
        ds_ij = p_ij * (dp_ij - D_i[:, None])
        dq_i += tl.dot(ds_ij, k_j, input_precision='ieee') # (Br, BLOCK_SIZE_D)
    
    dq_i = dq_i * scale

    tl.store(dq_block_ptr_dq, dq_i[None, :, :], boundary_check=(0, 1, 2))

import torch
import os

def check_tensor_gpu_ready(tensors):
    for t in tensors:
        assert t.is_contiguous(), 'A tensor is not contiguous.'
        if not os.environ.get('TRITON_INTERPRET') == '1':
            assert t.is_cuda, 'A tensor is not on cuda device.'

class ScaledDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, scale=None):
        B, H, T, d = q.shape # this is the same as how the sdpa in torch is implemented
        d_k, d_v = k.shape[-1], v.shape[-1]
        assert d_k == d and d_v == d, 'the input matrices should have the same d_model'
        check_tensor_gpu_ready((q, k, v))
        
        legal_d = {16, 32, 64, 128, 256, 512}
        assert d in legal_d, f'unsupported value of head dim, only supprt one of {legal_d}'

        q, k, v = q.view(B*H, T, d), k.view(B*H, T, d), v.view(B*H, T, d)
        scale = 1 / d ** 0.5 if scale == None else scale

        # output tensor
        o = torch.empty_like(v) 

        # save L for backward
        L = torch.empty((B*H, T), device=q.device, dtype=torch.float32) 

        Br, Bc = 16, 16 # Use smaller blocks to reduce shared memory pressure on the GPU
        # Br, Bc = 32, 32
        # use this condition to simplify the indexing of tiles on the diagonal
        # when causal=True
        assert Br % Bc == 0, 'keep Br % Bc == 0 to make kernel index simple'
        grid = lambda meta: (triton.cdiv(T, meta['Br']), B*H, 1)
        att_fwd_kernel[grid](
            scale, q, k, v, o, L, 
            q.stride(0), q.stride(1), q.stride(2), # k, v, o has the same stride
            L.stride(0), L.stride(1), 
            B*H, T, d,
            is_causal, 
            BLOCK_SIZE_D = triton.next_power_of_2(d),
            Br = Br, Bc = Bc,
        )
        
        ctx.save_for_backward(q, k, v, L, o)
        ctx.Br = Br
        ctx.Bc = Bc
        ctx.scale = scale
        ctx.is_causal = is_causal

        return o.view(B, H, T, d)
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, L, o = ctx.saved_tensors # shape of q, k, v, o is (B*H, T, d)
        Br, Bc    = ctx.Br, ctx.Bc
        scale     = ctx.scale
        is_causal = ctx.is_causal
        # print(is_causal)

        B, H, T, d = do.shape
        do = do.view(B*H, T, d)

        # ---------- Phase 1: compute D = sum(o * do, axis=-1)
        D = torch.zeros_like(L)
        grid = lambda meta: (triton.cdiv(T, meta['Br']), B*H, 1)
        att_bwd_pre_kernel[grid](
            o, do, D,
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1), 
            B*H, T, d,
            BLOCK_SIZE_D=triton.next_power_of_2(d), 
            Br=Br, Bc=Bc,       
        )

        # ---------- Phase 2: compute dk, dv
        dk = torch.zeros_like(do) # shape is (B*H, T, d)
        dq = torch.zeros_like(do) 
        dv = torch.zeros_like(do)

        stage = 1 if is_causal else 0 # do not pass bool into kernel, easy to cause error
                                      # if the bool argument 'is_causal' is passed
                                      # and used in the kernel, there would be an compiler error
                                      # but this problem do not happen in dq kernel
        Multi_factor = Br // Bc

        grid_kv = lambda meta: (triton.cdiv(T, meta['Bc']), B*H, 1)
        att_bwd_dkdv[grid_kv](
            q, k, v, dk, dv, 
            do, D, L, 
            q.stride(0), q.stride(1), q.stride(2), 
            D.stride(0), D.stride(1), 
            B*H, T, d,
            scale, 
            Multi_factor, stage, 
            BLOCK_SIZE_D=triton.next_power_of_2(d),
            Br=Br, Bc=Bc,  
        )


        grid_q = lambda meta: (triton.cdiv(T, meta['Br']), B*H, 1)
        att_bwd_dq[grid_q](
            q, k, v, dq, 
            do, D, L, 
            q.stride(0), q.stride(1), q.stride(2), 
            D.stride(0), D.stride(1), 
            B*H, T, d,
            scale, is_causal,
            BLOCK_SIZE_D=triton.next_power_of_2(d),
            Br=Br, Bc=Bc,  
        )

        return dq.view((B, H, T, d)), dk.view((B, H, T, d)), dv.view((B, H, T, d)), None, None






def main():
    torch.manual_seed(0)
    import torch.nn.functional as F

    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    sdpa = ScaledDotProductAttention.apply
    B, H, T, d = 1, 1, 2048, 16
    for causal in [False]:
        shape = (B, H, T, d)
        q = torch.ones(shape, requires_grad=True, device=DEVICE, dtype=torch.float32)
        k = torch.ones(shape, requires_grad=True, device=DEVICE, dtype=torch.float32)
        v = torch.arange(B*H*T*d, device=DEVICE, dtype=torch.float32).reshape(shape)
        v.requires_grad_(True)
        do = torch.ones(shape, device=DEVICE, dtype=torch.float32)
                
        # triton result
        o = sdpa(q, k, v, causal, 1.0)
        o.backward(do)

        triton_dq, triton_dk, triton_dv = [g.grad.clone() for g in (q, k, v)] 
        for g in (q, k, v): g.grad = None # clear grad

        # torch result
        torch_o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=1.0)
        torch_o.backward(do)
        torch_dq, torch_dk, torch_dv = [g.grad.clone() for g in (q, k, v)]
        for g in (q, k, v): g.grad = None # clear grad

        # compare
        print('o diverge:', torch.max(torch.abs(torch_o - o)).item())
        print('dq diverge:', torch.max(torch.abs(torch_dq - triton_dq)).item())
        print('dk diverge:', torch.max(torch.abs(torch_dk - triton_dk)).item())
        print('dv diverge:', torch.max(torch.abs(torch_dv - triton_dv)).item())
        assert torch.allclose(torch_o , o , atol=1e-2)
        assert torch.allclose(torch_dq, triton_dq, atol=1e-2)
        assert torch.allclose(torch_dk, triton_dk, atol=1e-2)
        assert torch.allclose(torch_dv, triton_dv, atol=1e-2)
        print('='*50)

if __name__ == '__main__':
    main()