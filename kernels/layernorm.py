import triton
import triton.language as tl

@triton.jit
def layernorm_fwd_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, eps, 
    mean_ptr, rstd_ptr, 
    N, 
    stride_x_row, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    The implementation of the forward part is the same as triton tutorial.
    """
    idx_row = tl.program_id(0)
    x_row_ptr = x_ptr + idx_row * stride_x_row
    y_row_ptr = y_ptr + idx_row * stride_x_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x_ptrs = x_row_ptr + offsets
    y_ptrs = y_row_ptr + offsets
    w_ptrs = weight_ptr + offsets
    b_ptrs = bias_ptr + offsets

    x_f32 = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w_f32 = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
    b_f32 = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)    
    
    mean = tl.sum(x_f32, axis=0) / N
    # var = tl.sum(x_f32 * x_f32) / N - mean * mean # Caution: do not use this expression
    # Can suffer catastrophic cancellation when mean and individual x are large but the variance is small
    x_centered = tl.where(mask, x_f32 - mean, 0.0)
    var = tl.sum(x_centered * x_centered) / N
    rstd = tl.rsqrt(var + eps)
    y_f32 = (x_f32 - mean) * rstd * w_f32 + b_f32  

    tl.store(mean_ptr + idx_row, mean)
    tl.store(rstd_ptr + idx_row, rstd)
    tl.store(y_ptrs, y_f32, mask=mask)   # would cast y_f32 back to the type of y
                                         # original type info of y is in y_ptrs


# The hard part of backward kernel is computing dw and db.
# Each row in x has its part of contribution to dw and db.
# Each row is processed in a PI and gets its partial of dw and db. 
# The final dw/db is accumulated across all rows(PIs).
# It's inefficient if all parallel PIs writes to the same address.
# All parallel writes would become sequential if they target the same memory location,
# causing severe performance degradation due to memory contention and serialization.

# An efficient way to implement the backward pass is as following:
# 1. split all rows into multiple groups (e.g., GROUP_SIZE groups), where each group
#    will accumulate its own partial dw/db independently.
# 2. put consecutive rows in different groups, because consecutive rows would 
#    write their part of partial dw/db at almost the same time. if they need to 
#    write to the same address, the parallel execution would become sequential again.
#    By distributing them across groups, we avoid this contention.
#    Example: row 0 → group 0, row 1 → group 1, ..., row k → group k%GROUP_SIZE
# 3. Each PI processes its assigned row and accumulates its contribution to the 
#    corresponding group's partial dw/db. Since consecutive PIs write to different
#    group buffers, writes remain parallel.
# 4. After all rows are processed, perform a final reduction across groups to merge
#    the partial dw/db values into the final gradients. This is typically done in
#    a separate kernel or a final reduction step with fewer threads.
#    Final dw = sum(dw_group_0, dw_group_1, ..., dw_group_K)
#    Final db = sum(db_group_0, db_group_1, ..., db_group_K)

@triton.jit
def layernorm_bwd_dx_kernel(
    x_ptr, w_ptr, dy_ptr, 
    m_ptr, r_ptr,
    dx_ptr, group_ptr, locks_ptr, # locks_ptr is unused here, just for compatibility
    N, stride_row, 
    group_M, group_stride_m, group_stride_n, # should be 2*BLOCK_SIZE_N
    BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_N: tl.constexpr,
):
    """
    The atomic add operation is much slow than the way triton tutorial is used. 
    this inefficient implementation is just for illustration. use the 'wo_atomic'
    version below.
    """
    idx_row = tl.program_id(0)
    row_ptr = idx_row * stride_row
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N

    x_ptrs  = x_ptr  + row_ptr + offsets
    dy_ptrs = dy_ptr + row_ptr + offsets
    w_ptrs  = w_ptr + offsets

    # convert to float32
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    w  = tl.load(w_ptrs , mask=mask, other=0.0).to(tl.float32)
    wdy = w * dy
    wdy = tl.where(mask, wdy, 0.0)
    scalar_1 = tl.sum(wdy) / N

    x  = tl.load(x_ptrs , mask=mask, other=0.0).to(tl.float32)
    m  = tl.load(m_ptr + idx_row) 
    r  = tl.load(r_ptr + idx_row)
    hat_x = (x - m) * r
    hat_x = tl.where(mask, hat_x, 0.0)
    scalar_2 = tl.sum(wdy * hat_x) / N

    dx_ptrs = dx_ptr + row_ptr + offsets
    dx = (wdy - scalar_1 - scalar_2 * hat_x) * r
    tl.store(dx_ptrs, dx, mask=mask)
    
    # for dw, db
    partial_dw = (hat_x * dy).to(w.dtype)
    partial_db = dy.to(w.dtype)

    id_group = idx_row % group_M
    group_start = group_ptr + id_group * group_stride_m
    offsets_dw = tl.arange(0, BLOCK_SIZE_N)
    offsets_db = tl.arange(0, BLOCK_SIZE_N) + BLOCK_SIZE_N
    group_ptrs_dw = group_start + offsets_dw * group_stride_n
    group_ptrs_db = group_start + offsets_db * group_stride_n
    tl.atomic_add(group_ptrs_dw, partial_dw, mask=mask)
    tl.atomic_add(group_ptrs_db, partial_db, mask=mask)

@triton.jit
def layernorm_bwd_dx_wo_atomic_ops_kernel(
    x_ptr, w_ptr, dy_ptr, 
    m_ptr, r_ptr,
    dx_ptr, group_ptr, locks_ptr, 
    N, stride_row, 
    group_M, group_stride_m, group_stride_n, # should be 2*BLOCK_SIZE_N
    BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_N: tl.constexpr,
):
    """
    The implementation is different from the triton tutorial that
    we store partial dw and db together in 'group'. And so each PI would
    load and store partial dw and db at consecutive address. This could improve
    coalescing.
    """
    idx_row = tl.program_id(0)
    row_ptr = idx_row * stride_row
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N

    x_ptrs  = x_ptr  + row_ptr + offsets
    dy_ptrs = dy_ptr + row_ptr + offsets
    w_ptrs  = w_ptr + offsets

    # convert to float32
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    w  = tl.load(w_ptrs , mask=mask, other=0.0).to(tl.float32)
    wdy = w * dy
    wdy = tl.where(mask, wdy, 0.0)
    scalar_1 = tl.sum(wdy) / N

    x  = tl.load(x_ptrs , mask=mask, other=0.0).to(tl.float32)
    m  = tl.load(m_ptr + idx_row) 
    r  = tl.load(r_ptr + idx_row)
    hat_x = (x - m) * r
    hat_x = tl.where(mask, hat_x, 0.0)
    scalar_2 = tl.sum(wdy * hat_x) / N

    dx_ptrs = dx_ptr + row_ptr + offsets
    dx = (wdy - scalar_1 - scalar_2 * hat_x) * r
    tl.store(dx_ptrs, dx, mask=mask)
    
    # for dw, db
    partial_dw = (hat_x * dy).to(w.dtype)
    partial_db = dy.to(w.dtype)

    id_group = idx_row % group_M    # locks has 2 * group_M elements
    lock_ptr = locks_ptr + id_group # save lock data in the 1st half of locks
    count_ptr = lock_ptr + group_M  # save count data in the 2nd half of locks

    # locate the address for patial dw/db
    group_start = group_ptr + id_group * group_stride_m
    offsets_dw = tl.arange(0, BLOCK_SIZE_N)
    offsets_db = tl.arange(0, BLOCK_SIZE_N) + BLOCK_SIZE_N
    group_ptrs_dw = group_start + offsets_dw * group_stride_n
    group_ptrs_db = group_start + offsets_db * group_stride_n

    while tl.atomic_cas(lock_ptr, 0, 1) == 1: pass
    count = tl.load(count_ptr)
    if count == 0: # First store doesn't accumulate
        tl.atomic_xchg(count_ptr, 1)
    else:
        partial_dw += tl.load(group_ptrs_dw, mask=mask)
        partial_db += tl.load(group_ptrs_db, mask=mask)
    tl.store(group_ptrs_dw, partial_dw, mask=mask)
    tl.store(group_ptrs_db, partial_db, mask=mask)

    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)


@triton.jit
def layernorm_bwd_merge_dwdb_kernel(
    group_ptr, dwdb_ptr, 
    group_M, group_N, 
    group_stride_m, group_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Parallelly merge the partial dwdb into one row. 
    The 1st half of the data in the row is dw, the 2nd half is db.
    Each PI merge several (BLOCK_N) number of columns.
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    dwdb_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, group_M, BLOCK_M):
        rows = i + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < group_M) & (cols[None, :] < group_N)
        offsets = rows[:, None] * group_stride_m + cols[None, :] * group_stride_n
        dwdb_block += tl.load(group_ptr + offsets, mask=mask, other=0.0)
    sum_dwdb = tl.sum(dwdb_block, axis=0)
    tl.store(dwdb_ptr + cols, sum_dwdb, mask=cols < group_N)