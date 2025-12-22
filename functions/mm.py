import torch
import triton
import triton.language as tl
from kernels.mm_2D_naive import naive_mm_kernel_use_block_ptr
from kernels.mm_2D import official_mm_kernel, grouped_mm_2D_kernel

def mm(a, b, block_m=64, block_n=64, block_k=64, group_size=8, kernel_name='group'):

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous" # no need to check b.is_contiguous
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    if kernel_name == 'naive':
        active_kernel = naive_mm_kernel_use_block_ptr
        # use 2D grid, and so the index of pids is 2D
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    elif kernel_name == 'group':
        active_kernel = grouped_mm_2D_kernel
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    elif kernel_name == 'official':
        active_kernel = official_mm_kernel
        # use 1D grid, and so the index of pids is 1D
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )
        active_kernel[grid](
                a, b, c,  #
                M, N, K,  #
                a.stride(0), a.stride(1),  #
                b.stride(0), b.stride(1),  #
                c.stride(0), c.stride(1),  #
            )
        return c
    else:
        raise ValueError(f'wrong kernel_name, only support: group, naive, official')

    active_kernel[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            BLOCK_M=block_m, 
            BLOCK_N=block_n, 
            BLOCK_K=block_k,  #
            GROUP_SIZE=group_size,  #
        )
    return c


#### simple test

def test_mm(x, w, discrption: str):
    y_mm_naive = mm(x, w, kernel_name='naive')
    y_mm_group = mm(x, w, kernel_name='group')
    y_mm_official = mm(x, w, kernel_name='official')

    y_torch = x @ w
    print('//'* 6, discrption)
    print('Torch == Group:',torch.allclose(y_torch, y_mm_group), torch.max(torch.abs(y_torch - y_mm_group)).item())
    print('Torch == Naive:', torch.allclose(y_torch, y_mm_naive), torch.max(torch.abs(y_torch - y_mm_naive)).item())
    print('Torch == Official:', torch.allclose(y_torch, y_mm_official), torch.max(torch.abs(y_torch - y_mm_official)).item())
    
def main(args):
    torch.manual_seed(37)
    for M, K, N in [(512, 512, 512), (4, 10, 20), (1000, 1000, 1000)]:
        print(f'M={M}, K={K}, N={N}:')
        x = torch.ones((M, K), dtype=torch.bfloat16, device='cuda')
        w = torch.ones((K, N), dtype=torch.bfloat16, device='cuda')
        test_mm(x, w, discrption='interger test:')

        x = torch.rand((M, K), dtype=torch.bfloat16, device='cuda')
        w = torch.rand((K, N), dtype=torch.bfloat16, device='cuda')
        test_mm(x, w, discrption='float test:')
        print('\n')

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_name', default='group', help='choose which kernel to use')
    args = parser.parse_args()
    main(args)

# # Check if error is uniform or localized
# error = torch.abs(triton_y - torch_y)
# print("Max error:", error.max())
# print("Mean error:", error.mean())
# print("Error std:", error.std())
# print("Number of exact matches:", (error == 0).sum().item(), "out of", error.numel())

# # Check specific locations
# print("\nFirst element - Torch:", torch_y[0, 0], "Triton:", triton_y[0, 0])
# print("Last element - Torch:", torch_y[-1, -1], "Triton:", triton_y[-1, -1])

# # Check where errors occur
# print("\nError map shape:", error.shape)
# print("Errors > 0.01:", (error > 0.01).sum().item())


# command line: python -m functions.MatrixMultiple