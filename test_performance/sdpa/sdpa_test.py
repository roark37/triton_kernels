# import os
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported
                                     # when the datatype is bfloat16, do not use interpret mode
import torch
import triton
import torch.nn.functional as F
import sys
sys.path.append('/home/roark/Documents/7_cuda/triton/rk/rk_triton')
sys.path.append('/home/roark/Documents/7_cuda/triton')
from functions.SDPA import ScaledDotProductAttention

sdpa = ScaledDotProductAttention.apply

def test_fwd(configs, mode:str='ones', dtype=torch.bfloat16, device=torch.device('cuda')):
    """
    Test fwd function of sdpa
    
    Args: 
    mode(str): choose the value of inputs. 
               - 'ones': q and k would be torch.ones, v would be torch.arange(0, B*H*T*d)
               - 'random': all q, k, v would be init use torch.rand
    """
    B, H = 1, 1
    for causal in configs['causal']:
        for scale in configs['scale']:
            for T in configs['T']:
                for d in configs['d']:
                    shape = (B*H, T, d)
                    if mode == 'ones':
                        q, k = torch.ones((2, B*H, T, d), device=device, dtype=dtype)
                        v = torch.arange(B*H*T*d, device=device, dtype=dtype).reshape(shape)
                    elif mode == 'random':
                        q, k, v = torch.rand((3, B*H, T, d), device=device, dtype=dtype)
                    else:
                        raise ValueError('Wrong type of mode!')
                    
                    o = sdpa(q, k, v, causal)
                    torch_o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

                    if not torch.allclose(o, torch_o):
                        dist = torch.max(torch.abs(o - torch_o)).item()
                        print(f'causal={causal}, T={T}, d={d}: {dist}')
                        # for i in range(v.shape[2]):
                        #     if not torch.allclose(o[0, 0, i], torch_o[0, 0, i]):
                        #         # print(i, end='/')
                        #         if i in [0]:
                        #             print(i)
                        #             print(f'triton {i}th row: {o[0, 0, i]}')
                        #             print(f'torch  {i}th row: {torch_o[0, 0, i]}')
                        #             print(f'dist of {i}th row: {o[0, 0, i] - torch_o[0, 0, i]}')
                        # print('/n')
                        print('='*80)
                    else:
                        print(f'causal={causal}, T={T}, d={d}: pass')
                        print('='*80)

def test_bwd(configs, mode:str='random', dtype=torch.bfloat16, device=torch.device('cuda')):
    B, H = 1, 1
    for causal in configs['causal']:
        for scale in configs['scale']:
            for T in configs['T']:
                for d in configs['d']:
                    shape = (B*H, T, d)
                    if mode == 'ones':
                        q = torch.ones(shape, requires_grad=True, device=device, dtype=dtype)
                        k = torch.ones(shape, requires_grad=True, device=device, dtype=dtype)
                        v = torch.arange(B*H*T*d, device=device, dtype=dtype).reshape(shape)
                        v.requires_grad_(True)
                    elif mode == 'random':
                        q = torch.rand(shape, requires_grad=True, device=device, dtype=dtype)
                        k = torch.rand(shape, requires_grad=True, device=device, dtype=dtype)
                        v = torch.rand(shape, requires_grad=True, device=device, dtype=dtype)
                    else:
                        raise ValueError('Wrong type of mode!')
        
                    do = torch.ones(shape, device=device, dtype=dtype)
                            
                    # triton result
                    o = sdpa(q, k, v, causal)
                    o.backward(do)

                    triton_dq, triton_dk, triton_dv = [g.grad.clone() for g in (q, k, v)] 
                    for g in (q, k, v): g.grad.zero_() # clear grad

                    # torch result
                    torch_o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                    torch_o.backward(do)
                    torch_dq, torch_dk, torch_dv = [g.grad.clone() for g in (q, k, v)]
                    for g in (q, k, v): g.grad.zero_() # clear grad

                    # compare
                    print(torch.max(torch.abs(torch_o - o)).item())
                    print(torch.max(torch.abs(torch_dq - triton_dq)).item())
                    print(torch.max(torch.abs(torch_dk - triton_dk)).item())
                    print(torch.max(torch.abs(torch_dv - triton_dv)).item())
                    assert torch.allclose(torch_o , o , atol=1e-2)
                    assert torch.allclose(torch_dq, triton_dq, atol=1e-2)
                    assert torch.allclose(torch_dk, triton_dk, atol=1e-2)
                    assert torch.allclose(torch_dv, triton_dv, atol=1e-2)
                    print('='*50)


if __name__ == '__main__':
    configs = {
        'causal': [True], 
        'scale':  [1.0], 
        'T':      [256], # [2 ** i for i in range(10, 15)], 
        'd':      [128], 
        'tf32':   False,
    }

    # configs = {
    #     'causal': [False], 
    #     'scale': [1.0] , 
    #     'T': [i for i in range(100, 400, 5)], 
    #     'd': [256], 
    # }

    torch.manual_seed(0)
    # test_fwd(configs, mode='random', dtype=torch.bfloat16) 
    # Problem Recording: 
    # if mode='ones', dtype=torch.float32, scale=1.0, T>= 2048, d=16, then there is a problem in forward kernel.
    # 'o_i = adjust_factor[:, None] * o_i + acc' this line gives wrong answer. See the code part for explaination.

    
    test_bwd(configs, dtype=torch.bfloat16)