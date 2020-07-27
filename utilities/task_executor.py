import torch
from torch.multiprocessing import Pool, set_sharing_strategy
from concurrent.futures import as_completed
import concurrent.futures

set_sharing_strategy('file_system')

def execute(f, args, pbar=None):
    results = []
    for arg in args:
        result = f(arg)
        results.append(result)
        if pbar is not None : pbar.update()
    return results
        
def concurrent_execute(f, args, mode, num_workers=1, pbar=None):
    if mode == 'multiprocess': return execute_multiprocess(f, args, num_workers, pbar)
    if mode == 'multithread': return execute_multithread(f, args, num_workers, pbar)

def execute_multithread(f, args, num_workers, pbar=None):
    results = []
    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        for r in executor.map(f, args):
            results.append(r)
            if pbar is not None: pbar.update()
        executor.shutdown()
    return results

def execute_multiprocess(f, args, num_workers, pbar=None):
    results = []
    with Pool(processes=num_workers) as p:
        results = []
        for result in p.imap(f, args):
            results.append(result)
            if pbar is not None: pbar.update()
        p.close()
    return results