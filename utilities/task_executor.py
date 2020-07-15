from multiprocessing import Pool

def execute(f, args, pbar=None):
    results = []
    for arg in args:
        result = f(arg)
        results.append(result)
        if pbar is not None : pbar.update()
    return results

def par_execute(f, args, num_processes=1, pbar=None):
    results = []
    with Pool(processes=num_processes) as p:
        for result in p.imap(f, args):
            results.append(result)
            if pbar is not None: pbar.update()
    return results
        