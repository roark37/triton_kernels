# import os
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import torch.nn.functional as F
import triton
import os
from functions.SDPA import ScaledDotProductAttention
from test_performance.performance_utils import time_cuda, profile


def torch_sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)    


def naive_sdpa(q, k, v, causal):
    T, d = q.shape[-2], q.shape[-1]
    attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)

    qkt = q @ k.transpose(-2, -1) / d ** 0.5
    if causal:
        temp_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    qkt += attn_bias
    score = torch.softmax(qkt, dim=-1)
    alpha = score @ v
    return alpha


def main():
    torch.manual_seed(0)
    import torch.nn.functional as F

    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    sdpa = ScaledDotProductAttention.apply

    # test
    # simple_test(sdpa, F.scaled_dot_product_attention, DEVICE)

    # profile
    B, H, T, d = 1, 8, 4096, 128
    dtype = torch.bfloat16
    shape = (B*H, T, d)
    q = torch.rand(shape, requires_grad=True, device=DEVICE, dtype=dtype)
    k = torch.rand(shape, requires_grad=True, device=DEVICE, dtype=dtype)
    v = torch.rand(shape, requires_grad=True, device=DEVICE, dtype=dtype)
    do = torch.ones(shape, device=DEVICE, dtype=dtype)

    # time_cuda(sdpa, (q, k, v, True, 1.0), do)
    
    functions = {'sdpa': sdpa, 'torch_sdpa': torch_sdpa, 'naive_sdpa': naive_sdpa}
    for name, func in functions.items():
        table = profile(func, (q, k, v, True), do=do, descript=name)
        print(table)

if __name__ == '__main__':
    main()