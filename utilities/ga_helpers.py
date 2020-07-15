import numpy as np
from utilities.task_executor import par_execute, execute
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.fitness_evaluator import FitnessEvaluator
from typing import List
from tqdm import tqdm

def calculate_fitnesses(populations: List[NetworkGenotype], evaluator: FitnessEvaluator, num_workers=None, gen=None):
    eval = evaluator.eval_fitness
    desc = f'Generation {gen}' if gen is not None else ''
    with tqdm(total=len(populations), desc=desc) as pbar:
        fitnesses = par_execute(eval, populations, num_workers, pbar) if num_workers is not None else execute(eval, populations, pbar)
        fitnesses = np.array(fitnesses)
        pbar.set_postfix(max_f=fitnesses.max(), min_f=fitnesses.min(), avg_f=fitnesses.mean())
    return np.array(fitnesses)
    