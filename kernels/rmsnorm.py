import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_fused_kernel(
    x_ptr, y_ptr, inv_rms_ptr, 
    w_ptr, eps,
    N, stride_row, 
    BLOCK_SIZE: tl.constexpr,
):
    idx_row = tl.program_id(0)

    # offsets
    row_dist = idx_row * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # load data
    x = tl.load(x_ptr + row_dist + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    # computation
    N_f32 = N.to(tl.float32)
    eps_f32 = eps.to(tl.float32)
    inv_rms_x = tl.rsqrt(tl.sum(x * x) / N_f32 + eps_f32)

    y = x * inv_rms_x * w

    # store result
    tl.store(y_ptr + row_dist + offsets, y, mask=mask)
    tl.store(inv_rms_ptr + idx_row, inv_rms_x)


@triton.jit
def rmsnorm_bwd_dx_kernel(
    dout_ptr, x_ptr, 
    inv_rms_ptr, w_ptr,
    dx_ptr, group_ptr, locks_ptr,
    N, stride_row, 
    group_size_m, group_stride_m, group_stride_n, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    similar to layernorm
    """
    idx_row = tl.program_id(0)
    row_dist = idx_row * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x  = tl.load(x_ptr    + row_dist + offsets, mask=mask, other=0.0)
    dy = tl.load(dout_ptr + row_dist + offsets, mask=mask, other=0.0)
    inv_rms  = tl.load(inv_rms_ptr + idx_row)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    wdy = w * dy
    normed_x = x * inv_rms
    scalar = tl.sum(wdy * normed_x) / N.to(tl.float32)
    dx = inv_rms * (wdy - scalar * normed_x)

    tl.store(dx_ptr + row_dist + offsets, dx, mask=mask)

    # for parital dw
    normed_x = x * inv_rms
    partial_dw = (normed_x * dy).to(w.dtype)
    id_group = idx_row % group_size_m

    lock_ptr = locks_ptr + id_group # save lock data in the 1st half of locks
    count_ptr = lock_ptr + group_size_m  # save count data in the 2nd half of locks

    # locate the address for patial dw/db
    group_start = group_ptr + id_group * group_stride_m
    offsets_dw = tl.arange(0, BLOCK_SIZE)
    group_ptrs_dw = group_start + offsets_dw * group_stride_n

    while tl.atomic_cas(lock_ptr, 0, 1) == 1: pass
    count = tl.load(count_ptr)
    if count == 0: # First store doesn't accumulate
        tl.atomic_xchg(count_ptr, 1)
    else:
        partial_dw += tl.load(group_ptrs_dw, mask=mask)
    tl.store(group_ptrs_dw, partial_dw, mask=mask)

    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)


@triton.jit
def rmsnorm_bwd_merge_dw_kernel(
    group_ptr, dw_ptr, 
    group_size_m, group_size_n,
    group_stride_m, group_stride_n, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):

    """
    Parallelly merge the partial dw into one row. 
    The 1st half of the data in the row is dw, the 2nd half is db.
    Each PI merge several (BLOCK_N) number of columns.
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    dw_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, group_size_m, BLOCK_M):
        rows = i + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < group_size_m) & (cols[None, :] < group_size_n)
        offsets = rows[:, None] * group_stride_m + cols[None, :] * group_stride_n
        dw_block += tl.load(group_ptr + offsets, mask=mask, other=0.0)
    sum_dw = tl.sum(dw_block, axis=0)
    tl.store(dw_ptr + cols, sum_dw, mask=cols < group_size_n)
