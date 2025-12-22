import torch
import triton
import sys
sys.path.append('/home/roark/Documents/7_cuda')
import os

from triton_tutorials.layernorm import LayerNorm
from modules.LayerNorm import LayerNormFused

from torch.profiler import profile, ProfilerActivity


DEVICE = triton.runtime.driver.active.get_active_torch_device()
ln = LayerNormFused.apply
official_ln = LayerNorm.apply


# define inputs
shape = (4096, 512 * 20)
x  = torch.rand(shape, dtype=torch.float32, device=DEVICE, requires_grad=True)
w  = torch.ones((x.shape[-1], ), device=DEVICE, requires_grad=True)
b  = torch.zeros((x.shape[-1], ), device=DEVICE, requires_grad=True)
dy = .1 * torch.rand(shape, dtype=torch.float32, device=DEVICE)


act = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
sort_by_keyword = 'cpu_time_total'

with profile(
    activities=act, 
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
) as pf_tri:
    y = ln(x, w, b)
    y.backward(dy)
    torch.cuda.synchronize()

with profile(
    activities=act, 
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
) as pf_tt:
    off_y = official_ln(x, w.shape, w, b, 1e-6)
    off_y.backward(dy)
    torch.cuda.synchronize()

# pf_tri.export_chrome_trace('tri_layernorm.json')
# pf_tt.export_chrome_trace('tt_layernorm.json')

# book results
os.makedirs('profile_results', exist_ok=True)
table_file = 'profile_results/table2.txt'
with open(table_file, mode='w') as f:
    f.write(pf_tri.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    f.write(pf_tt.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
