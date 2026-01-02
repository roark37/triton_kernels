import torch
import triton
import os
from kernels.sdpa import flash_fwd_kernel, flash_bwd_pre_kernel, flash_bwd_dkdv, flash_bwd_dq

def check_tensor_gpu_ready(tensors):
    for t in tensors:
        assert t.is_contiguous(), 'A tensor is not contiguous.'
        if not os.environ.get('TRITON_INTERPRET') == '1':
            assert t.is_cuda, 'A tensor is not on cuda device.'

class ScaledDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        BH, Tq, d = q.shape # this is the same as how the sdpa in torch is implemented
        Tk = k.shape[-2]

        d_k, d_v = k.shape[-1], v.shape[-1]
        assert d_k == d and d_v == d, 'the input matrices should have the same d_model'
        check_tensor_gpu_ready((q, k, v))
        
        legal_d = {16, 32, 64, 128, 256, 512}
        assert d in legal_d, f'unsupported value of head dim, only supprt one of {legal_d}'
        
        scale = 1 / (d ** 0.5)

        BF16 = True if q.dtype == torch.bfloat16 else False

        # output tensor
        o = torch.empty_like(v) 

        # save L for backward
        L = torch.empty((BH, Tq), device=q.device, dtype=torch.float32) 

        # larger outer blocks could redule number of iterations in the inner loop, 
        # and so the total loading times of the matrices loaded in the inner loop.
        # But it also increase shared memory pressure on the GPU
        # so the best strategy is use the largest block the gpu could afford
        B_outerloop, B_innerloop = 64, 32
        # B_outerloop, B_innerloop = 32, 32

        assert Tq % B_innerloop == 0 and Tq % B_outerloop == 0, \
               'T must be divisible by B_innerloop and B_outerloop. ' 
        # 1. The 'tl.make_block_ptr()' can not pad 'float('-inf')' when loading data.
        #    If T is not divisible by column block size, and using 'tl.make_block_ptr()' 
        #    with zero padding, the value of m_i would be twisted when loading the last 
        #    block of k.
        #    This also twists the value of softmax(qkT) under non-causal mode.
        #    We could correct it by adding a mask after the last block of k is loaded.
        #    see 'att_fwd_inner' part of code for details.

        #    To avoid adding complex and inefficient masking logic (), a simple solution is
        #    letting T to be divisible by Bc. This sidesteps the padding issue entirely.
        
        # 2. because the forward kernel use B_innerloop as the culumn block size,
        #    but the backward kernel for dkdv use B_outerloop as the culumn block size,
        #    so, we need both Tq % B_innerloop == 0 and Tq % B_outerloop == 0 be meet.

        # 3. Zero padding on rows does not distort the final result.

        # 4. zero padding has no affect to the result of softmax when causal=True.

        grid = lambda meta: (triton.cdiv(Tq, meta['Q_TILE_SIZE']), BH)
        flash_fwd_kernel[grid](
            q, k, v, 
            o, L, 
            q.stride(0), q.stride(1), q.stride(2), # k, v, o has the same stride
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1), 
            Tq, Tk, 
            scale,
            BF16,
            is_causal=is_causal, 
            D = d,
            Q_TILE_SIZE = B_outerloop, K_TILE_SIZE = B_innerloop,
        )
        
        ctx.save_for_backward(q, k, v, L, o)
        ctx.B_outerloop = B_outerloop
        ctx.B_innerloop = B_innerloop
        ctx.scale = scale
        ctx.is_causal = is_causal

        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, L, o = ctx.saved_tensors # shape of q, k, v, o is (B*H, T, d)
        B_outerloop, B_innerloop = ctx.B_outerloop, ctx.B_innerloop
        scale     = ctx.scale
        is_causal = ctx.is_causal

        BF16 = True if q.dtype == torch.bfloat16 else False

        BH, T, d = do.shape

        # ---------- Phase 1: compute D = sum(o * do, axis=-1)
        D = torch.zeros_like(L)
        grid = lambda meta: (triton.cdiv(T, meta['B_outerloop']), BH)
        flash_bwd_pre_kernel[grid](
            o, do, D,
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1), 
            T, d, 
            B_outerloop=B_outerloop, B_innerloop=B_innerloop,       
        )

        # print(torch.max(torch.abs(D.to(do.dtype) - torch.sum(o * do, dim = -1))))

        # ---------- Phase 2: compute dk, dv
        dk = torch.zeros_like(do) # shape is (B*H, T, d)
        dq = torch.zeros_like(do) 
        dv = torch.zeros_like(do)

        grid_kv = lambda meta: (triton.cdiv(T, meta['B_outerloop']), BH)
        flash_bwd_dkdv[grid_kv](
            q, k, v, dk, dv, 
            do, D, L, 
            q.stride(0), q.stride(1), q.stride(2), 
            D.stride(0), D.stride(1), 
            T, d, BF16,
            scale, is_causal, 
            BLOCK_SIZE_D=triton.next_power_of_2(d),
            B_outerloop=B_outerloop, B_innerloop=B_innerloop,       
        )

        grid_q = lambda meta: (triton.cdiv(T, meta['B_outerloop']), BH)
        flash_bwd_dq[grid_q](
            q, k, v, dq, 
            do, D, L, 
            q.stride(0), q.stride(1), q.stride(2), 
            D.stride(0), D.stride(1), 
            T, d, BF16,
            scale, is_causal,
            BLOCK_SIZE_D=triton.next_power_of_2(d),
            B_outerloop=B_outerloop, B_innerloop=B_innerloop,       
        )

        return dq, dk, dv, None, None


## Test code
import torch.nn.functional as F
triton_sdpa = ScaledDotProductAttention.apply

def compare_tensor(x, x_torch, dtype, description=None):
    tol = {
        torch.float32: 1e-4,
        torch.float16: 3e-3,
        torch.bfloat16: 1e-2,
    }[dtype]

    diff = (x - x_torch).abs()
    max_err = diff.max().item()
    print(f"{description}: Max error: {max_err:.6e}")

    # assert max_err < tol, f"FAILED! error={max_err}"
    # if max_err > tol:
    #     for i in range(T):
    #         print((O_ref[:,:,i] - O_torch[:,:,i]).abs().max().item(), end='| ')
    #     print('\n')

def run_test(T=256, B=2, H=4, d=128, causal=False, dtype=torch.float32, device="cuda"):
    torch.manual_seed(0)

    Q = torch.randn(B*H, T, d, requires_grad=True, device=device, dtype=dtype)
    K = torch.randn(B*H, T, d, requires_grad=True, device=device, dtype=dtype)
    V = torch.randn(B*H, T, d, requires_grad=True, device=device, dtype=dtype)
    dO = torch.ones(B*H, T, d, device=device, dtype=dtype)

    # FA2 reference

    O_ref = triton_sdpa(Q, K, V, causal)

    O_ref.backward(dO)
    dq, dk, dv = Q.grad.clone(), K.grad.clone(), V.grad.clone()

    for i in (Q, K, V):
        i.grad.zero_()

    # PyTorch SDP
    O_torch = F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=causal,
    )
    O_torch.backward(dO)
    dq_torch, dk_torch, dv_torch = Q.grad.clone(), K.grad.clone(), V.grad.clone()

    d_triton = [dq, dk, dv]
    d_torch = [dq_torch, dk_torch, dv_torch]

    print('-' * 60)
    print(f'[dtype={dtype}, causal={causal}]')
    compare_tensor(O_ref, O_torch, dtype, description='output')
    for i, (x, x_torch) in enumerate(zip(d_triton, d_torch)):
        if torch.is_tensor(x) and x is not None:
            description = ['Q', 'K', 'V']
            compare_tensor(x, x_torch, dtype, description=description[i])
    

if __name__ == '__main__':

    # Run several tests
    for dtype in [torch.float32, torch.bfloat16]:
    # for dtype in [torch.bfloat16]:
        for causal in [True, False]:
            run_test(T=1024, B=1, H=1, d=64, causal=causal, dtype=dtype)

# commandline: python -m functions.SDPA