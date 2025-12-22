import numpy as np
import torch
from torch.profiler import ProfilerActivity, record_function
from typing import Callable, Tuple

# helper function
def if_cuda_then_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# timing
def time_cuda(func, inputs, do, warm_up=2, num_trials=5):
    for _ in range(warm_up):
        o = func(*inputs)
        o.backward(do)
        for i in inputs:
            if torch.is_tensor(i):
                i.grad.zero_()
    if torch.cuda.is_available(): torch.cuda.synchronize()

    times_fwd, times_bwd = [], []
    for _ in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        fwd_event = torch.cuda.Event(enable_timing=True)
        bwd_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        o = func(*inputs)
        fwd_event.record()

        o.backward(do)
        bwd_event.record()
        
        # ensures the GPU has finished works before we ask for the elapsed time.
        torch.cuda.synchronize()

        for i in inputs:
            if torch.is_tensor(i):
                i.grad.zero_()

        times_fwd.append(start_event.elapsed_time(fwd_event))  # already in ms
        times_bwd.append(fwd_event.elapsed_time(bwd_event))    # already in ms
    
    fwd_mean, fwd_std = np.mean(times_fwd), np.std(times_fwd)
    bwd_mean, bwd_std = np.mean(times_bwd), np.std(times_bwd)
    print(f'forward mean: {fwd_mean:.2f}ms, backward mean: {bwd_mean:.2f}ms')

# torch profile
def profile(func: Callable, inputs: Tuple, do=None, warm_up=1, profile_memory:bool=False, descript: str='func'):
    for _ in range(warm_up):
        func(*inputs)
    if_cuda_then_sync()

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        profile_memory=profile_memory,
    ) as prof:
        # use record_function decorator to label arbitrary code ranges with customized name
        with record_function(descript): 
            o = func(*inputs)
            if_cuda_then_sync()
        
            if do is not None:
                o.backward(do)
                if_cuda_then_sync()
            
            for i in inputs:
                if torch.is_tensor(i) and i.grad is not None: 
                    i.grad.zero_()
    
    table = prof.key_averages().table(
        sort_by='cuda_time_total', max_name_column_width=30,row_limit=20
    )

    return table


