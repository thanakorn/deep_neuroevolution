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
    def __init__(self, num_populations, fitness_evaluator, selection_pressure, mutation_prob, mutation_power, crossover_prob,
                 model_type: M, **model_params):
        self.num_populations = num_populations
        self.fitness_evaluator = fitness_evaluator
        self.selection_pressure = selection_pressure
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        self.croosover_prob = crossover_prob
        self.model_type = model_type
        self.model_params = model_params
    
    def run(self, num_generations):
        populations = [NetworkGranularGenotype(self.model_type.genetic_schema(**self.model_params)) for i in range(self.num_populations)]
        fitnesses = np.zeros(self.num_populations)
        for gen in range(num_generations):
            with tqdm(total=len(populations), desc=f'Generation {gen+1}') as t:
                for i, p in enumerate(populations):
                    fitnesses[i] = self.fitness_evaluator.eval_fitness(self.model_type, p)
                    t.update()
                t.set_postfix(max_f=fitnesses.max(), min_f=fitnesses.min(), avg_f=fitnesses.mean())
            
            best = populations[np.argmax(fitnesses)]
            new_gen = self.new_generation(populations, fitnesses)
            populations = new_gen
        
        return best
    
    def new_generation(self, olg_gen, fitnesses):
        raise NotImplementedError()

class SimpleGA(GeneticAlgorithm):     
    def new_generation(self, old_gen, fitnesses):
        num_elites = int(self.selection_pressure * self.num_populations)
        elites = select_elites(old_gen, fitnesses, num_elites)
        new_generation = [elites[-1]] # Best model survives and is carried to next gen
        mutation_populations = gen_population_mutation(elites, n=self.num_populations - 1, mutation_rate=self.mutation_prob, mutation_power=self.mutation_power)
        # crossover_populations = gen_population_crossover(elites, n=int(self.num_populations / 2))
        new_generation.extend(mutation_populations)
        # new_generation.extend(crossover_populations)
        return new_generation