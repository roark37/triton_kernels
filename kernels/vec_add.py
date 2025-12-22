import torch
from torch import tensor
import triton
import triton.language as tl



@triton.jit
def vec_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for element-wise vector addition.
    
    This kernel performs parallel element-wise addition of two vectors using 
    Triton's GPU programming model. Each program instance processes a block 
    of elements.
    
    Args:
        a_ptr: Pointer to the first input tensor in device memory
        b_ptr: Pointer to the second input tensor in device memory
        c_ptr: Pointer to the output tensor in device memory
        n: Total number of elements in the vectors
        BLOCK_SIZE: Number of elements processed per program instance (compile-time constant)
    
    Notes:
        - Uses masking to handle cases where n is not divisible by BLOCK_SIZE
        - Each program instance (identified by program_id) processes one contiguous block
        - Memory accesses are coalesced for optimal performance
    """
    # locate starting point
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # get offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # load data
    a_offsets = a_ptr + offsets
    b_offsets = b_ptr + offsets
    c_offsets = c_ptr + offsets

    mask=offsets < n
    block_a = tl.load(a_offsets, mask)
    block_b = tl.load(b_offsets, mask)
    block_add = tl.add(block_a, block_b)
    tl.store(c_offsets, block_add, mask)



# api
def vec_add(a:tensor, b:tensor, block_size:int=128):
    """
    Perform element-wise addition of two tensors using Triton.

    Args:
        a: First input tensor (any shape)
        b: Second input tensor (must match shape of a)
        block_size: Number of elements processed per thread block (default: 128)
                    Must be divisible by 32 for optimal warp utilization
    
    Returns:
        A new tensor containing the element-wise sum of a and b
    
    Raises:
        AssertionError: If a and b have different shapes or block_size is not divisible by 32
    
    Example:
        >>> a = torch.randn(1000, device='cuda')
        >>> b = torch.randn(1000, device='cuda')
        >>> c = vec_add(a, b)
        >>> assert torch.allclose(c, a + b)
    
    Notes:
        - The grid is automatically sized to cover all elements
        - Uses ceiling division to ensure all elements are processed
        - Output tensor is allocated with the same properties as input a
    """

    assert a.shape == b.shape, 'a and b should have the same length.'
    assert block_size % 32 == 0, 'block_size must be divisible by 32.'

    len = a.numel()
    # grid = lambda meta: (tl.cdiv(len, meta['BLOCK_SIZE']), ) # got an Error!
    grid = lambda meta: (triton.cdiv(len, meta['BLOCK_SIZE']), )
    ## rk's Note
    #  - 'tl.cdiv' is a compile time ceiling function used in kernel
    #  - 'triton.cdiv' is a python level utility function used in host code

    c = torch.empty_like(a)
    vec_add_kernel[grid](a, b, c, len, BLOCK_SIZE=block_size)
    return c

