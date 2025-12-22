import torch
import torch.nn.functional as F
import triton
from kernels.layernorm import layernorm_fwd_kernel, layernorm_bwd_merge_dwdb_kernel, \
                              layernorm_bwd_dx_kernel, layernorm_bwd_dx_wo_atomic_ops_kernel
from torch.profiler import record_function
import torch.cuda.nvtx as nvtx


MAX_BLOCK_MEMORY = 65536

class LayerNormFused(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight_shape, weight, bias, eps=1e-8):
        with record_function("LN::FP"):

            x_2D = x.reshape(-1, x.shape[-1])
            y_2D = torch.empty_like(x_2D)
            M, N = x_2D.shape

            mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
            rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
            # _var = torch.empty((M, ), dtype=torch.float32, device=x.device) # for debug

            # heuristics for block_size
            block_size_n = triton.next_power_of_2(N)
            max_block_size = MAX_BLOCK_MEMORY // x.element_size()  # 65536=64KB
            assert block_size_n <= max_block_size, 'fused kernel does not supports feature dim > 64KB'

            # heuristics for number of warps
            num_warps = min(max(block_size_n//256, 1), 8) # if block_size < 256, then 1
                                                        # if block_size > 2048, then 8
                                                        # else, block_size // 256
            grid = (M, )
            nvtx.range_push('fp')
            layernorm_fwd_kernel[grid](
                    x_2D, y_2D, weight, bias, eps,
                    mean, rstd, #_var,
                    N, 
                    x_2D.stride(0), 
                    BLOCK_SIZE=block_size_n, 
                    num_warps=num_warps,
                )
            nvtx.range_pop()

            ctx.save_for_backward(x_2D, weight, bias, mean, rstd)
            ctx.eps = eps
            ctx.block_size = block_size_n
            ctx.num_warps = num_warps

            return y_2D.reshape_as(x)#, mean, rstd, _var
    
    @staticmethod
    def backward(ctx, dy):
        with record_function("LN::BP"):
            dy_2D = dy.reshape(-1, dy.shape[-1])
            dx_2D = torch.empty_like(dy_2D)
            
            M, N = dy_2D.shape
            x_2D, w, b, mean, rstd = ctx.saved_tensors

            group_size_m = 64
            if N <= 8192: group_size_m = 96
            if N <= 4096: group_size_m = 128
            if N <= 1024: group_size_m = 256
            block_size_n = ctx.block_size
            group_size_n = 2 * block_size_n
            group = torch.zeros((group_size_m, group_size_n), dtype=torch.float32, device=w.device)
            locks = torch.zeros((2 * group_size_m, ), dtype=torch.int32, device=w.device)

            nvtx.range_push('dx')
            layernorm_bwd_dx_wo_atomic_ops_kernel[(M, )](
                        x_2D, w, dy_2D, 
                        mean, rstd,
                        dx_2D, group, locks,
                        N, x_2D.stride(0), 
                        group_size_m, group.stride(0), group.stride(1), 
                        BLOCK_SIZE_N=block_size_n, GROUP_SIZE_N=group_size_n,
                        num_warps=ctx.num_warps,
                    )
            nvtx.range_pop()

            dwdb = torch.zeros((group_size_n, ), dtype=w.dtype, device=w.device)
            block_m, block_n = 32, 128 # heuristic
            grid = lambda meta: (triton.cdiv(group_size_n, meta['BLOCK_N']), )
            nvtx.range_push('dw')
            layernorm_bwd_merge_dwdb_kernel[grid](
                        group, dwdb, 
                        group_size_m, group_size_n,
                        group.stride(0), group.stride(1),
                        BLOCK_M=block_m, BLOCK_N=block_n,
                        num_warps=ctx.num_warps,
                    )            
            nvtx.range_pop()
        
        dw, db = dwdb[:N], dwdb[block_size_n:block_size_n+N]
        # dw/db is accumulated in fp32; cast back to weight dtype
        if dw.dtype != w.dtype: dw = dw.to(w.dtype)
        if db.dtype != b.dtype: db = db.to(b.dtype)
        return dx_2D.reshape_as(dy), None, dw, db, None

def main():
    device = 'cuda'
    dtype = torch.float32
    torch.manual_seed(0)
    ln = LayerNormFused.apply

    M, N = 4096, 17408
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    eps=1e-8
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)

    for _ in range(3):
        with torch.no_grad():
            for k in [x, weight, bias]:
                k.grad = None
                
        y = ln(x, w_shape, weight, bias, eps)
        torch_y = F.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
        # print(torch.max(torch.abs(y - torch_y)).item())
        # print(torch.allclose(y, torch_y))
        # print(y)
        nvtx.range_push('bp')
        y.backward(dy)
        nvtx.range_pop()

if __name__ == '__main__':
    main()