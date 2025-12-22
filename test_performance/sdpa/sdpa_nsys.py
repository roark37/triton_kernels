import torch
import triton
import torch.nn.functional as F
from functions.SDPA import ScaledDotProductAttention
import torch.cuda.nvtx as nvtx


sdpa = ScaledDotProductAttention.apply
DEVICE = triton.runtime.driver.active.get_active_torch_device()
# sample run
# used when profiling using nsys

def run(func, bwd_active=False, device=DEVICE, dtype=torch.bfloat16):
    # print('Start running ...')

    B, H = 4, 4
    for T in [4096, 8192]:
        for d in [256]:
            shape = (B, H, T, d)
            q, k, v = torch.rand((3, B, H, T, d), device=device, dtype=dtype)
            
            nvtx.range_push('fwd'+str(T))
            o = func(q, k, v, True, 1.0)
            nvtx.range_pop()
            print(o.shape, o)

            if bwd_active:
                do = torch.ones(shape, device=device, dtype=dtype)
                nvtx.range_push('bwd'+str(T))
                o.backward(do)

                nvtx.range_pop()

                for g in (q, k, v): 
                    g.grad = None # clear grad

def torch_sdpa(q, k, v, causal, scale):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)    

def main():
    for func in [sdpa, torch_sdpa]:
        nvtx.range_push('triton')
        run(func)
        nvtx.range_pop()
        
        if torch.cuda.is_available(): torch.cuda.synchronize()


if __name__ == '__main__':
    main()