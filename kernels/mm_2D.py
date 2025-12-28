import triton
import triton.language as tl
#os.environ["TRITON_INTERPRET"] = "1" # for debugging

@triton.jit
def grouped_mm_2D_kernel(
    x_ptr, w_ptr, y_ptr, M, N, K,
    stride_x_m, stride_x_k,
    stride_w_k, stride_w_n,
    stride_y_m, stride_y_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, # how many blocks along the m(row) direction per group
    BF: tl.constexpr, # if BF is 1, the dtype is bfloat16, elif BF is 0, dtype is float32
):
    """
        - use two dimensional pids. 
        - The groupping is only in rows. Each GROUP_SIZE number of row blocks are groupped together.
        - The last group might has less than GROUP_SIZE row blocks
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_blocks_m = tl.num_programs(0)
    num_blocks_n = tl.num_programs(1)

    # get the pid_m number of the first block in the group
    id_group = pid_m // GROUP_SIZE
    id_pid_m_at_group_start = id_group * GROUP_SIZE

    # group size of the last group != GROUP_SIZE if (num_pid_m % GROUP_SIZE != 0)
    group_size_actual = min(GROUP_SIZE, (num_blocks_m - id_pid_m_at_group_start))

    # map from pid to the block of C the pid is going to process
    # need to expand the 2D index (pid_m, pid_n) to local index in a group
    # otherwise, the mapping relationship is very complex when: num_blocks_n % GROUP_SIZE != 0
    local_id_in_group_1D = pid_m % GROUP_SIZE * num_blocks_n + pid_n
    id_data_block_m = id_pid_m_at_group_start + local_id_in_group_1D % group_size_actual
    id_data_block_n = local_id_in_group_1D // group_size_actual

    # # Compute offsets for this block
    # offsets_m = id_data_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offsets_n = id_data_block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # offsets_k = tl.arange(0, BLOCK_K)
    
    # x_ptrs = x_ptr + offsets_m[:, None] * stride_x_m + offsets_k[None, :] * stride_x_k
    # w_ptrs = w_ptr + offsets_k[:, None] * stride_w_k + offsets_n[None, :] * stride_w_n

    x_block_ptr = tl.make_block_ptr(
        x_ptr, shape=(M, K), strides=(stride_x_m, stride_x_k), 
        offsets=(id_data_block_m * BLOCK_M, 0), 
        block_shape=(BLOCK_M, BLOCK_K), 
        order=(0, 1), 
    )
    w_block_ptr = tl.make_block_ptr(
        w_ptr, shape=(K, N), strides=(stride_w_k, stride_w_n),
        offsets=(0, id_data_block_n * BLOCK_N), 
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),   
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr, shape=(M, N), strides=(stride_y_m, stride_y_n),
        offsets=(id_data_block_m * BLOCK_M, id_data_block_n * BLOCK_N), 
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),   
    )
    # Accumulate result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # x = tl.load(x_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_K, other=0.0)
        # w = tl.load(w_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_K, other=0.0)
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        w = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option='zero')
        
        if BF == 1: 
            acc = tl.dot(x, w, acc)
        else:
            acc = tl.dot(x, w, acc, input_precision="ieee")
        # x_ptrs += BLOCK_K * stride_x_k
        # w_ptrs += BLOCK_K * stride_w_k
        x_block_ptr = x_block_ptr.advance((0, BLOCK_K))
        w_block_ptr = w_block_ptr.advance((BLOCK_K, 0))
    
    # Store result
    if BF == 1: acc = acc.to(dtype=tl.bfloat16)
    # y_ptrs = y_ptr + offsets_m[:, None] * stride_y_m + offsets_n[None, :] * stride_y_n
    # mask_y = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    # tl.store(y_ptrs, acc, mask=mask_y)
    tl.store(y_block_ptr, acc, boundary_check=(0, 1))

autotune_configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE': 8}, num_warps=16, num_stages=4), 
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE': 8}, num_warps=4, num_stages=2), 
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE': 8}, num_warps=8, num_stages=2), 
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_SIZE': 8}, num_warps=8, num_stages=4), 
]
@triton.autotune(
    configs=autotune_configs,
    key=['M', 'N', 'K']
)
@triton.jit
def official_mm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        GROUP_SIZE: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    
    use one dimensional pids
    """

    pid = tl.program_id(axis=0)

    # number of pids on m and n dim
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # the index of group the pid is in
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group

    # get the pid number of the first block in the group along m direction
    id_pid_m_at_group_start = group_id * GROUP_SIZE

    # group size of the last group != GROUP_SIZE if (num_pid_m % GROUP_SIZE != 0)
    group_size_m = min(num_pid_m - id_pid_m_at_group_start, GROUP_SIZE)

    # map from pid to the block of C the pid is going to process
    local_id_in_group_1D = pid % num_pid_in_group
    pid_m = id_pid_m_at_group_start + (local_id_in_group_1D % group_size_m)
    pid_n = local_id_in_group_1D // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        accumulator = tl.dot(a, b, accumulator, input_precision='ieee')

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # `accumulator` is converted back to fp16 after the loop.
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)