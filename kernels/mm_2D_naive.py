import triton
import triton.language as tl

@triton.jit
def naive_mm_kernel_use_ptr_arithmetic(
    x_ptr, w_ptr, y_ptr, 
    M, N, K, 
    stride_x_m, stride_x_k,
    stride_w_k, stride_w_n,
    stride_y_m, stride_y_n,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, # do not delete this to keep the signature the same as the grouped kernel           
    BF: tl.constexpr,
):
    # locate start point
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    offsets_m = start_m + tl.arange(0, BLOCK_M)
    offsets_n = start_n + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offsets_m[:, None] * stride_x_m + offsets_k[None, :] * stride_x_k
    w_ptrs = w_ptr + offsets_k[:, None] * stride_w_k + offsets_n[None, :] * stride_w_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):

        # - load data
        x = tl.load(x_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_K, other=0.0) # pointer arithmetic way
        w = tl.load(w_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_K, other=0.0)

        # - computation
        # acc = tl.dot(x, w, acc, allow_tf32=False) # set allow_tf32=False explicitly is deprecated
        if BF == 1: 
            acc = tl.dot(x, w, acc) # tf32 would not activated on non-float32 dtype
        else:
            acc = tl.dot(x, w, acc, input_precision="ieee") # do not use tf32 for float32 type of data

        # - update pointer position
        x_ptrs += BLOCK_K * stride_x_k # pointer arithmetic way
        w_ptrs += BLOCK_K * stride_w_k
    if BF == 1: acc = acc.to(dtype=tl.bfloat16)

    # y_ptrs = y_ptr + offsets_m[:, None] * N + offsets_n[None, :]
    y_ptrs = y_ptr + offsets_m[:, None] * stride_y_m + offsets_n[None, :] * stride_y_n
    mask_y = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=mask_y)

@triton.jit
def naive_mm_kernel_use_block_ptr(
    x_ptr, w_ptr, y_ptr, 
    M, N, K, 
    stride_x_m, stride_x_k,
    stride_w_k, stride_w_n,
    stride_y_m, stride_y_n,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, # do not delete this to keep the signature the same as the grouped kernel
    BF: tl.constexpr, 
):
    """
    Matrix multiplication of two tensors: c = a @ b
    - the naive mm method computes the blocks in the output matrix c by
    following the row-major ordering. 
    - each program access block_m rows in M and block_n columns in N to 
      compute a (block_m, block_n) shaped tile in c. 

    Eg: If there are 16(4x4) block in c, and the pid can be 2D(used in our code)
        or equivalent 1D format. as follows:

        pids in 2D format:                  pids in 1D format:
        -----------------------------       ---------------------
        | 0, 0 | 0, 1 | 0, 2 | 0, 3 |       |  0 |  1 |  2 |  3 | 
        -----------------------------       ---------------------
        | 1, 0 | 1, 1 | 1, 2 | 1, 3 |       |  4 |  5 |  6 |  7 |  
        -----------------------------  <=>  ---------------------
        | 2, 0 | 2, 1 | 2, 2 | 2, 3 |       |  8 |  9 | 10 | 11 |      
        -----------------------------       ---------------------
        | 3, 0 | 3, 1 | 3, 2 | 3, 3 |       | 12 | 13 | 14 | 15 |  
        -----------------------------       ---------------------
        
        The naive method for computing mm launches blocks in the same order as the
        1D pids array.
    """
    # locate start point
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    # prepare block pointers
    x_block_ptr = tl.make_block_ptr(
        x_ptr, shape=(M, K), strides=(stride_x_m, stride_x_k),
        offsets=(pid_m * BLOCK_M, 0), 
        block_shape=(BLOCK_M, BLOCK_K), 
        order=(0, 1),                   # 0,1-> first along row, then along column
    )
    w_block_ptr = tl.make_block_ptr(
        w_ptr, shape=(K, N), strides=(stride_w_k, stride_w_n),
        offsets=(0, pid_n * BLOCK_N), 
        block_shape=(BLOCK_K, BLOCK_N), 
        order=(1, 0),                   # 1,0-> first along column, then along row
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr, shape=(M, N), strides=(stride_y_m, stride_y_n),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), 
        block_shape=(BLOCK_M, BLOCK_N), 
        order=(0, 1),                   # 0,1-> first along row, then along column
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=2):

        # - load data
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        w = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # - computation
        # acc = tl.dot(x, w, acc, allow_tf32=False) # set allow_tf32=False explicitly
        if BF == 1: 
            acc = tl.dot(x, w, acc) # tf32 would not activated on non-float32 dtype
        else:
            acc = tl.dot(x, w, acc, input_precision="ieee") # do not use tf32 for float32 type of data


        # - update pointer position
        x_block_ptr = x_block_ptr.advance((0, BLOCK_K))
        w_block_ptr = w_block_ptr.advance((BLOCK_K, 0))
    if BF == 1: acc = acc.to(dtype=tl.bfloat16)
    tl.store(y_block_ptr, acc, boundary_check=(0, 1))

