import torch
import ray
import concurrent.futures
from ray.util.multiprocessing import Pool
from concurrent.futures import as_completed, ThreadPoolExecutor

global PROCESS_POOL
global THREAD_POOL
PROCESS_POOL = None
THREAD_POOL = None

def execute(f, args, pbar=None):
    results = []
    for arg in args:
        result = f(arg)
        results.append(result)
        if pbar is not None : pbar.update()
    return results
        
def concurrent_execute(f, args, mode, num_workers=1, pbar=None):
    if mode == 'multiprocess': return execute_multiprocess(f, args, num_workers=num_workers, pbar=pbar)
    if mode == 'multithread': return execute_multithread(f, args, num_workers=num_workers, pbar=pbar)

def execute_multithread(f, args, num_workers, pbar=None):
    global THREAD_POOL
    THREAD_POOL = ThreadPoolExecutor(num_workers) if THREAD_POOL is None else THREAD_POOL
    p = THREAD_POOL
    results = []
    for r in p.map(f, args):
        results.append(r)
        if pbar is not None: pbar.update()
    return results

def execute_multiprocess(f, args, num_workers, pbar=None):
    global PROCESS_POOL
    PROCESS_POOL = Pool(processes=num_workers) if PROCESS_POOL is None else PROCESS_POOL
    p = PROCESS_POOL
    results = []
    for result in p.starmap(f, args, chunksize=num_workers):
        results.append(result)
        if pbar is not None: pbar.update()
    return results