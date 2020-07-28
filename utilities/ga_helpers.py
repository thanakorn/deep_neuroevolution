import numpy as np
import ray
from utilities.task_executor import execute, concurrent_execute
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.fitness_evaluator import FitnessEvaluator
from typing import List
from tqdm import tqdm

def calculate_fitnesses(populations: List[NetworkGenotype], evaluator: FitnessEvaluator, gen=None, num_workers=None, run_mode=None):
    eval = evaluator.eval_fitness
    desc = f'Generation {gen}' if gen is not None else ''
    with tqdm(total=len(populations), desc=desc) as pbar:
        fitnesses = concurrent_execute(eval, populations, run_mode, num_workers, pbar) if num_workers is not None else execute(eval, populations, pbar)
        fitnesses = np.array(fitnesses)
        pbar.set_postfix(max_f=fitnesses.max(), min_f=fitnesses.min(), avg_f=fitnesses.mean())
    return np.array(fitnesses)

def calculate_fitnesses_distributed(gen, populations: List[NetworkGenotype], evaluator, **kwargs):
    
    def to_iterator(futures):
        while futures:
            done, futures = ray.wait(futures)
            yield ray.get(done[0])
    
    desc = f'Generation {gen}' if gen is not None else ''
    
    remote_evaluators = [evaluator.remote(**kwargs) for p in populations]
    result_futures = [e.eval_fitness.remote(p) for (e, p) in zip(remote_evaluators, populations)]
    fitnesses = []
    
    # for r in tqdm(to_iterator(result_futures), total=len(result_futures)):
    #     fitnesses.append(r)
    with tqdm(total=len(result_futures), desc=desc) as pbar:
        for r in to_iterator(result_futures):
            fitnesses.append(r)
            pbar.update()
        fitnesses = np.array(fitnesses)
        pbar.set_postfix(max_f=fitnesses.max(), min_f=fitnesses.min(), avg_f=fitnesses.mean())
    
    return fitnesses
    
    