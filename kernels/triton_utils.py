# utils functions
import os
import torch

def check_tensor_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous(), 'A tensor is not contiguous.'
        if not os.environ.get('TRITON_INTERPRET') == '1': 
            assert t.is_cuda, 'A tensor is not on cuda device.'