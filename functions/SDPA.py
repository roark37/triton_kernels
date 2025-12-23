# import os
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import torch.nn.functional as F
import triton
import os
from kernels.sdpa import att_fwd_kernel, att_bwd_pre_kernel, att_bwd_dkdv, att_bwd_dq
from kernels.sdpa_bf16 import att_fwd_kernel_bf16


def check_tensor_gpu_ready(tensors):
    for t in tensors:
        assert t.is_contiguous(), 'A tensor is not contiguous.'
        if not os.environ.get('TRITON_INTERPRET') == '1':
            assert t.is_cuda, 'A tensor is not on cuda device.'

class ScaledDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, scale=None):
        B, H, T, d = q.shape # this is the same as how the sdpa in torch is implemented
        d_k, d_v = k.shape[-1], v.shape[-1]
        assert d_k == d and d_v == d, 'the input matrices should have the same d_model'
        check_tensor_gpu_ready((q, k, v))
        
        legal_d = {16, 32, 64, 128, 256, 512}
        assert d in legal_d, f'unsupported value of head dim, only supprt one of {legal_d}'
        
        q, k, v = q.view(B*H, T, d), k.view(B*H, T, d), v.view(B*H, T, d)
        scale = 1 / (d ** 0.5) if scale == None else scale

        BF16 = True if q.dtype == torch.bfloat16 else False

        # output tensor
        o = torch.empty_like(v) 

        # save L for backward
        L = torch.empty((B*H, T), device=q.device, dtype=torch.float32) 

        # larger blocks could redule number of iterations in the inner loop, and so
        # reduce the total memory cost. but also increase shared memory pressure on the GPU
        # so the best strategy is use the largest block the gpu could afford
        Br, Bc = 32, 32
        # Br, Bc = 16, 16

        assert T % Bc == 0, 'T must be divisible by Bc. ' 
        # The 'tl.make_block_ptr()' can not pad 'float('-inf')' when loading data.
        # If T is not divisible by Bc, and using 'tl.make_block_ptr()' with zero
        # padding, the value of m_i would be twisted when loading the last block of k.
        # This also twists the value of softmax(qkT) under non-causal mode.
        # We could correct it by adding a mask after the last block of k is loaded.
        # see 'att_fwd_inner' part of code for details.

        # To avoid adding complex masking logic (which would be inefficient),
        # here we choose to restrict T to be divisible by Bc. 
        # This sidesteps the padding issue entirely.
        
        # And zero padding has no affect to the result of softmax when causal=True.

        grid = lambda meta: (triton.cdiv(T, meta['Br']), B*H, 1)
        att_fwd_kernel[grid](
            scale, q, k, v, o, L, 
            q.stride(0), q.stride(1), q.stride(2), # k, v, o has the same stride
            L.stride(0), L.stride(1), 
            B*H, T, d, 
            BF16,
            is_causal=is_causal, 
            BLOCK_SIZE_D = triton.next_power_of_2(d),
            Br = Br, Bc = Bc,
        )
        
        ctx.save_for_backward(q, k, v, L, o)
        ctx.Br = Br
        ctx.Bc = Bc
        ctx.scale = scale
        ctx.is_causal = is_causal

        return o.view(B, H, T, d)
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, L, o = ctx.saved_tensors # shape of q, k, v, o is (B*H, T, d)
        Br, Bc    = ctx.Br, ctx.Bc
        scale     = ctx.scale
        is_causal = ctx.is_causal

        BF16 = True if q.dtype == torch.bfloat16 else False

        B, H, T, d = do.shape
        do = do.view(B*H, T, d)

        # ---------- Phase 1: compute D = sum(o * do, axis=-1)
        D = torch.zeros_like(L)
        grid = lambda meta: (triton.cdiv(T, meta['Br']), B*H, 1)
        att_bwd_pre_kernel[grid](
            o, do, D,
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1), 
            B*H, T, d, 
            Br=Br, Bc=Bc,       
        )

        # ---------- Phase 2: compute dk, dv
        dk = torch.zeros_like(do) # shape is (B*H, T, d)
        dq = torch.zeros_like(do) 
        dv = torch.zeros_like(do)

        Multi_factor = Br // Bc

        grid_kv = lambda meta: (triton.cdiv(T, meta['Bc']), B*H, 1)
        att_bwd_dkdv[grid_kv](
            q, k, v, dk, dv, 
            do, D, L, 
            q.stride(0), q.stride(1), q.stride(2), 
            D.stride(0), D.stride(1), 
            B*H, T, d, BF16,
            scale, is_causal, 
            Multi_factor, 
            BLOCK_SIZE_D=triton.next_power_of_2(d),
            Br=Br, Bc=Bc,  
        )

        grid_q = lambda meta: (triton.cdiv(T, meta['Br']), B*H, 1)
        att_bwd_dq[grid_q](
            q, k, v, dq, 
            do, D, L, 
            q.stride(0), q.stride(1), q.stride(2), 
            D.stride(0), D.stride(1), 
            B*H, T, d, BF16,
            scale, is_causal,
            BLOCK_SIZE_D=triton.next_power_of_2(d),
            Br=Br, Bc=Bc,  
        )

        return dq.view((B, H, T, d)), dk.view((B, H, T, d)), dv.view((B, H, T, d)), None, None