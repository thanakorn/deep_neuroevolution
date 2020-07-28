import torch
import numpy as np
from genetic_algorithm.ga import GeneticAlgorithm, SimpleGA
from genetic_algorithm.opetators import *
from model.genetic_network import GeneticNetwork
from utilities.ga_helpers import calculate_fitnesses_distributed
from typing import TypeVar

M = TypeVar('M', GeneticNetwork, GeneticNetwork)

class DistributedGeneticAlgorithm(SimpleGA):
    def __init__(self, num_populations, fitness_evaluator, selection_pressure, mutation_prob, mutation_power, crossover_prob, **evaluator_args):
        super().__init__(num_populations, fitness_evaluator, selection_pressure, mutation_prob, mutation_power, crossover_prob)
        self.evaluator_args = evaluator_args
        
    def run(self, populations, num_generations, num_workers=None, run_mode=None):
        solution = None
        for gen in range(num_generations):
            fitnesses = calculate_fitnesses_distributed(gen, populations, self.fitness_evaluator, **self.evaluator_args)
            solution = populations[fitnesses.argmax()]
            new_gen = self.new_generation(populations, fitnesses)
            populations = new_gen
        return solution