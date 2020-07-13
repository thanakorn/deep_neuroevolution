import torch
import copy
import numpy as np
import concurrent.futures

from concurrent.futures import as_completed
from tqdm import tqdm, trange
from typing import List
from genetic_algorithm.genotype import *
from genetic_algorithm.opetators import *
from model.genetic_network import GeneticNetwork
from typing import TypeVar

M = TypeVar('M', GeneticNetwork, GeneticNetwork)

class GeneticAlgorithm():
    def __init__(self, num_populations, fitness_evaluator, selection_pressure, mutation_prob, mutation_power, crossover_prob):
        self.num_populations = num_populations
        self.fitness_evaluator = fitness_evaluator
        self.selection_pressure = selection_pressure
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        self.croosover_prob = crossover_prob
    
    def run(self, populations, num_generations, num_workers=1):
        solution = None
        for gen in range(num_generations):
            fitnesses = np.zeros(len(populations))
            with tqdm(total=len(populations), desc=f'Generation {gen+1}') as t:
                # fitness_futures = [executor.submit(self.fitness_evaluator.eval_fitness, p) for p in populations]
                # for i, f in zip(range(self.num_populations), as_completed(fitness_futures)):
                for i, p in enumerate(populations):
                    # fitnesses[i] = f.result()
                    fitnesses[i] = self.fitness_evaluator.eval_fitness(p)
                    t.update()
                solution = populations[np.argmax(fitnesses)]
                t.set_postfix(max_f=fitnesses.max(), min_f=fitnesses.min(), avg_f=fitnesses.mean())

            new_gen = self.new_generation(populations, fitnesses)
            populations = new_gen
        
        return solution
    
    def new_generation(self, olg_gen, fitnesses):
        raise NotImplementedError()

class SimpleGA(GeneticAlgorithm):     
    def new_generation(self, old_gen, fitnesses):
        num_elites = int(self.selection_pressure * self.num_populations)
        elites = select_elites(old_gen, fitnesses, num_elites)
        new_generation = [elites[-1]] # Best model survives and is carried to next gen
        mutation_populations = gen_population_mutation(elites, n=int(self.num_populations / 2) - 1, mutation_rate=self.mutation_prob, mutation_power=self.mutation_power)
        crossover_populations = gen_population_crossover(elites, n=int(self.num_populations / 2))
        new_generation.extend(mutation_populations)
        new_generation.extend(crossover_populations)
        return new_generation