import torch
from torch import tensor
from kernels.triton_utils import check_tensor_gpu_ready
import triton
from triton.runtime import driver
import triton.language as tl
import torch.cuda.nvtx as nvtx
# In this file, there are two softmax kernels implemented by using different method
# 1. The softmax() use persistent PIs, 
# 2. The softmax_per_row simply process matrix row by row.

# hardware attributes for kernel setting
DEVICE = triton.runtime.driver.active.get_active_torch_device()
properties = driver.active.utils.get_device_properties(DEVICE.index)
N_SM = properties['multiprocessor_count']
N_REGS = properties['max_num_regs']     # number of 32-bit regs on each SM
SIZE_SMEM = properties['max_shared_mem']# Maximum shared memory per PI = 99kb
WARP_SIZE = properties['warpSize']

@triton.jit
def softmax_kernel(
    x_ptr, y_ptr, 
    x_stride_m, x_stride_n,
    y_stride_m, y_stride_n,
    M, N, BLOCK_N: tl.constexpr, 
    num_stages: tl.constexpr, 
):
    row_start = tl.program_id(0)
    row_step_size = tl.num_programs(0)
    x_row_ptr = tl.make_block_ptr(
        x_ptr, 
        shape = (M, N),
        strides = (x_stride_m, x_stride_n), 
        offsets = (row_start, 0), 
        block_shape=(1, BLOCK_N),
        order=(1, 0), 
    )
    y_row_ptr = tl.make_block_ptr(
        y_ptr, 
        shape = (M, N),
        strides = (y_stride_m, y_stride_n), 
        offsets = (row_start, 0), 
        block_shape=(1, BLOCK_N),
        order=(1, 0), 
    )
    # iter through
    for _ in tl.range(0, tl.cdiv(M, row_step_size), num_stages=num_stages):
        # load one row
        # 注意，tl.load的padding_option只接受三种值：{“”, “zero”, “nan”}
        # 这里如果要用make_block_ptr的话，就要额外处理load出来的data
        # 否则就直接用标准的pointer arithmetic，此时other参数的取值没有限制
        x_row = tl.load(x_row_ptr, boundary_check=(0, 1), padding_option='zero')

        # create mask for valid elements
        mask = tl.arange(0, BLOCK_N) < N
        x_row = tl.where(mask, x_row, float('-inf')) # float('-inf') is a python float32 type literal
                                                       # not compatible when x is not torch.float32 type

        # compute for one row
        x_row = x_row - tl.max(x_row, axis=-1)
        x_row = tl.exp(x_row)
        y_row = x_row / tl.sum(x_row, axis=-1)
        # store result
        tl.store(y_row_ptr, y_row, boundary_check=(0, 1))
        # move the ptr to the next block(row)
        x_row_ptr = x_row_ptr.advance((row_step_size, 0))
        y_row_ptr = y_row_ptr.advance((row_step_size, 0))


def softmax(x:tensor, num_warps=32, num_stages=1):

    check_tensor_gpu_ready(x)
    x_2D = x.reshape(-1, x.shape[-1])
    y_2D = torch.empty_like(x_2D)

    M, N = x_2D.shape
    block_n = triton.next_power_of_2(N)

    kernel = softmax_kernel.warmup(
        x_2D, y_2D, 
        x_2D.stride(0), x_2D.stride(1), y_2D.stride(0), y_2D.stride(1),
        M, N, BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages, grid=(1, )
    )
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    max_pi_per_sm = N_REGS // (n_regs * WARP_SIZE * num_warps)
    pi_per_sm = min(max_pi_per_sm, SIZE_SMEM // size_smem)
    num_total_pi = min(N_SM * pi_per_sm, M)

    # print(f'num_PIs: {num_pi}, num_rows: {M}, rows_per_PI: {triton.cdiv(M, num_pi)}')

    grid = (num_total_pi, 1, 1)
    softmax_kernel[grid](
        x_2D, y_2D, 
        x_2D.stride(0), x_2D.stride(1), y_2D.stride(0), y_2D.stride(1),
        M, N, BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages,
    )

    return y_2D.reshape_as(x)

# 
@triton.jit
def softmax_per_row_kernel(x_ptr, sx_ptr, stride_m, BLOCK_N:tl.constexpr, stride_n=1):
    """
    Triton kernel for computing softmax along the last dimension of a 2D tensor.
    
    This kernel processes one row(one block) at a time in parallel. 
    Each PI handles a single row.
    
    Args:
        x_ptr: Pointer to the input tensor in global memory
        sx_ptr: Pointer to the output tensor in global memory
        stride_m: Number of elements per row (same as number of columns)
        BLOCK_N: Compile-time constant for block size (must be power of 2 >= stride_m)
        stride_n: Stride between consecutive elements (default=1 for contiguous tensors)
    
    Algorithm:
        1. Each program loads one row of data
        2. Compute max value for numerical stability
        3. Compute exp(x - max) for each element
        4. Normalize by sum to get softmax probabilities
    """

    pid = tl.program_id(0)
    block_start = pid * stride_m * stride_n

    # offsets: a list of pointers
    offsets = block_start + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < stride_m
    
    # load one block of x
    block_x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    # computation
    max_x = tl.max(block_x)
    exp_x = tl.exp(block_x - max_x)
    softmax_x = exp_x / tl.sum(exp_x)

    # store result
    tl.store(sx_ptr + offsets, softmax_x, mask=mask)

@triton.jit
def softmax_per_row_approx_kernel(x_ptr, sx_ptr, stride_m, BLOCK_N:tl.constexpr, stride_n=1):
    """
    Triton kernel for computing approximate softmax using exp2 instead of exp.
    
    This kernel is similar to softmax_2D_kernel but uses base-2 exponential (exp2)
    for faster computation at the cost of slight numerical accuracy. The conversion
    factor 1.44269504089 ≈ log2(e) converts from natural log to log base 2.
    
    Args:
        x_ptr: Pointer to the input tensor in global memory
        sx_ptr: Pointer to the output tensor in global memory
        stride_m: Number of elements per row (same as number of columns)
        BLOCK_N: Compile-time constant for block size (must be power of 2 >= stride_m)
        stride_n: Stride between consecutive elements (default=1 for contiguous tensors)
    
    Note:
        Uses exp2(x * log2(e)) as approximation for exp(x), which can be faster
        on some hardware but may have slightly different numerical properties.
    """

    pid = tl.program_id(0)
    block_start = pid * stride_m * stride_n

    # offsets
    offsets = block_start + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < stride_m
    
    # load one block of x
    block_x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    # computation
    max_x = tl.max(block_x)
    exp_x = tl.exp2((block_x - max_x) * 1.44269504089) # exp is computed using approx value
    softmax_x = exp_x / tl.sum(exp_x)

    # store result
    tl.store(sx_ptr + offsets, softmax_x, mask=mask)

def softmax_per_row(x:tensor, approx=False):
    """
        Compute row-wise softmax of a tensor using Triton.
        
        The computation is parallelized across rows, with each row processed independently
        by a separate program instance on the GPU.
        
        Args:
            x: Input tensor of shape (m, d). Must be 2D and contiguous in memory.
            approx: If True, uses exp2-based approximation for faster computation.
                    If False, uses standard exp for higher accuracy. Default is False.

        Example:
            >>> x = torch.randn(128, 512, device='cuda')
            >>> output = softmax(x)
            >>> assert torch.allclose(output.sum(dim=-1), torch.ones(128))
    """

    check_tensor_gpu_ready(x)
    x_2D = x.reshape(-1, x.shape[-1]) # convert to 2D shape
    sx = torch.empty_like(x_2D)       # create output

    # set program block
    m, d = x_2D.shape
    block_n = triton.next_power_of_2(d)
    grid = (m, )

    # launch kernel
    if approx:
        softmax_per_row_approx_kernel[grid](x, sx, stride_m=d, BLOCK_N=block_n)
    else:
        softmax_per_row_kernel[grid](x, sx, stride_m=d, BLOCK_N=block_n)

    return sx.reshape(x.shape)



def main(args):
    if args.per_row:
        sft = softmax_per_row
    else:
        sft = softmax

    N = 128
    for m in range(1000, 2000, 50):
        M = m  * 128
        x = torch.rand((M, N), device=DEVICE, dtype=torch.float32)

        nvtx.range_push('M'+str(M))
        y = sft(x)
        torch.cuda.synchronize()
        nvtx.range_pop()
        # print(y.shape)

    # print(torch.allclose(y, torch.softmax(x, axis=1)))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--per_row', action='store_true', 
                        help='use softmax_per_row_kernel')
    args = parser.parse_args()

    main(args)

# commandline: python -m kernels.softmax