import numpy as np
from utilities.task_executor import execute, concurrent_execute
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.fitness_evaluator import FitnessEvaluator
from typing import List
from tqdm import tqdm
from itertools import repeat

def calculate_fitnesses(populations: List[NetworkGenotype], evaluator: FitnessEvaluator, gen=None, num_workers=None, run_mode=None, *eval_args):
    eval = evaluator.eval_fitness
    eval_arguments = [(p, *args) for p, args in list(zip(populations, repeat(eval_args)))]
    desc = f'Generation {gen}' if gen is not None else ''
    with tqdm(total=len(populations), desc=desc) as pbar:
        fitnesses = concurrent_execute(eval, eval_arguments, mode=run_mode, num_workers=num_workers, pbar=pbar) if num_workers is not None else execute(eval, populations, pbar)
        fitnesses = np.array(fitnesses)
        pbar.set_postfix(max_f=fitnesses.max(), min_f=fitnesses.min(), avg_f=fitnesses.mean())
    return np.array(fitnesses)
    