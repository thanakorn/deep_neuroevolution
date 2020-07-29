import torch
import numpy as np

from genetic_algorithm.opetators import *
from model.genetic_network import GeneticNetwork
from utilities.ga_helpers import calculate_fitnesses
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
    
    def run(self, populations, num_generations, max_iterations=None, num_episodes_eval=1, num_workers=None, run_mode=None):
        solution = None
        for gen in range(num_generations):
            fitnesses = calculate_fitnesses(populations, self.fitness_evaluator, gen, num_workers, run_mode, max_iterations, num_episodes_eval)
            solution = populations[fitnesses.argmax()]
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