import torch
import triton
import torch.cuda.nvtx as nvtx
from kernels.rmsnorm import rmsnorm_fwd_fused_kernel, rmsnorm_bwd_dx_kernel, rmsnorm_bwd_merge_dw_kernel
                            
MAX_BLOCK_MEMORY = 65536 # Cap the memory cost for one block to be less than 64K

class RMSNormFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-8):
        """
        Performs RMSNorm on matrix `x` along the last dimension.

        The fused kernel supports feature dimensions up to 64 KB.  
        This limit exists because each program instance computes the RMSNorm 
        for a single row in one pass.

        If the feature dimension exceeds 64 KB, each row must be split into 
        multiple blocks, requiring additional 'for' loops in the kernel to 
        process these blocks. Due to the reduction operation (sum), these 
        loops introduce extra memory access overhead, which reduces the 
        efficiency gains from kernel fusion.
        """
        # reshape to 2D matrix
        x_2D = x.reshape(-1, x.shape[-1])
        if not x_2D.is_contiguous():
            x_2D = x_2D.contiguous()

        M, N = x_2D.shape
        # limit the memory cost for one block to be less than 64K
        MAX_BLOCK_SIZE = MAX_BLOCK_MEMORY // x.element_size() 
        block_n = triton.next_power_of_2(N)
        block_size = min(block_n, MAX_BLOCK_SIZE)

        assert block_size >= N, 'fused kernel only supports feature dim <= 64KB'

        # heuristics for number of warps
        num_warps = min(max(block_size//256, 1), 8) # if block_size < 256, then 1
                                                    # if block_size > 2048, then 8
                                                    # else, block_size // 256
        y_2D = torch.empty_like(x_2D)
        inv_rms_x = torch.empty(M, dtype=x.dtype, device=x.device)

        rmsnorm_fwd_fused_kernel[(M, )](
                x_2D, y_2D, inv_rms_x,
                weight, eps, 
                N, x_2D.stride(0), 
                BLOCK_SIZE = block_size,
                num_warps = num_warps,
            )
        ctx.block_size = block_size
        ctx.save_for_backward(x_2D, inv_rms_x, weight)
        ctx.num_warps = num_warps

        return y_2D.reshape_as(x)

    @staticmethod
    def backward(ctx, dout):
        x_2D, inv_rms_x, w = ctx.saved_tensors # inv_rms_x is 1/rms(x)
        BLOCK_SIZE = ctx.block_size
        M, N = x_2D.shape
        
        dout_2D = dout.reshape(-1, dout.shape[-1])
        if not dout_2D.is_contiguous():
            dout_2D = dout_2D.contiguous()
        
        dx_2D = torch.empty_like(dout_2D)

        group_size_m = 64
        if N <= 8192: group_size_m = 96
        if N <= 4096: group_size_m = 128
        if N <= 1024: group_size_m = 256
        group_size_n = BLOCK_SIZE
        partial_dw_group = torch.zeros((group_size_m, group_size_n), dtype=w.dtype, device=w.device)
        locks = torch.zeros((2 * group_size_m, ), dtype=torch.int32, device=w.device)

        rmsnorm_bwd_dx_kernel[(M, )](
                    dout_2D, x_2D, 
                    inv_rms_x, w,
                    dx_2D, partial_dw_group, locks,  
                    N, dout_2D.stride(0), 
                    group_size_m, partial_dw_group.stride(0), partial_dw_group.stride(1),
                    BLOCK_SIZE = BLOCK_SIZE,
                    num_warps = ctx.num_warps,
                )
            
        MAX_BLOCK_SIZE = MAX_BLOCK_MEMORY // dout_2D.element_size()
        block_m = triton.next_power_of_2(M)
        BLOCK_SIZE_DW = min(block_m, MAX_BLOCK_SIZE)
        assert BLOCK_SIZE_DW >= M, 'fused kernel only supports block for one column <= 64KB'
        
        dw = torch.zeros(N, dtype=w.dtype, device=w.device)
        block_m, block_n = 32, 128 # heuristic   
        grid = lambda meta: (triton.cdiv(group_size_n, meta['BLOCK_N']), )
        
        nvtx.range_push('dw')
        rmsnorm_bwd_merge_dw_kernel[grid](
            partial_dw_group, dw, 
            group_size_m, group_size_n,
            partial_dw_group.stride(0), partial_dw_group.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n,
            num_warps=ctx.num_warps,
        )            
        nvtx.range_pop()          

        # dw is accumulated in fp32; cast back to weight dtype
        if dw.dtype != w.dtype: dw = dw.to(w.dtype)
        return dx_2D.reshape_as(dout), dw, None # None for eps
        