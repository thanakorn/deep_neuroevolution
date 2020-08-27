import numpy as np
import random

from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm.opetators import *
from utilities.pareto_solution import find_pareto
from utilities.ga_helpers import calculate_fitnesses
from utilities.learning_info import LearningInfo

class MultiObjGA(GeneticAlgorithm):
    def __init__(self, num_populations, fitness_evaluators, mutation_prob, mutation_power):
        self.num_populations = num_populations
        self.fitness_evaluators = fitness_evaluators
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        
    def run(self, populations, num_generations, max_iterations=None, num_episodes_eval=1, visualize=False, num_workers=None, run_mode=None):
        avg_fitnesses, max_fitnesses = [], []
        solution = None
        for gen in range(num_generations):
            fitnesses = [calculate_fitnesses(populations, e, gen, num_workers, run_mode, max_iterations, num_episodes_eval, visualize) for e in self.fitness_evaluators]
            avg_fitnesses.append(fitnesses[0].mean())
            max_fitnesses.append(fitnesses[0].max())
            solution = populations[fitnesses[0].argmax()]
            new_gen = self.new_generation(populations, fitnesses)
            populations = new_gen
        
        return solution, LearningInfo(avg_fitnesses, max_fitnesses)
    
    def new_generation(self, old_gen, fitnesses):
        non_dominated_idx = find_pareto(fitnesses)
        elites = [old_gen[i] for i in non_dominated_idx]
        new_generation = [old_gen[np.argmax(fitnesses[0])]] # Best model survives
        if len(elites) < 2:
            mutation_populations = gen_population_mutation(elites, n=self.num_populations - 1, mutation_rate=self.mutation_prob, mutation_power=self.mutation_power)
            new_generation.extend(mutation_populations)
        else:
            mutation_populations = gen_population_mutation(elites, n=int(self.num_populations / 2) - 1, mutation_rate=self.mutation_prob, mutation_power=self.mutation_power)
            crossover_populations = gen_population_crossover(elites, n=int(self.num_populations / 2))
            new_generation.extend(mutation_populations)
            new_generation.extend(crossover_populations)
        
        random.shuffle(new_generation)
        return new_generation