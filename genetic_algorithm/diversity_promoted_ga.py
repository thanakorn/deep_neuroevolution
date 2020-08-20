import numpy as np
import random
from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm.opetators import *
from utilities.pareto_solution import find_pareto

class DiversityPromotedGA(GeneticAlgorithm):
    def __init__(self, num_populations, fitness_evaluator, diversity_evaluator, mutation_prob, mutation_power, crossover_prob):
        self.num_populations = num_populations
        self.fitness_evaluator = fitness_evaluator
        self.diversity_evaluator = diversity_evaluator
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        self.croosover_prob = crossover_prob
    
    def new_generation(self, old_gen, fitnesses, *args):
        diversity_scores = self.diversity_evaluator.eval_diversity(old_gen)
        non_dominated_idx = find_pareto([fitnesses, diversity_scores])
        elites = [old_gen[i] for i in non_dominated_idx]
        new_generation = [elites[np.argmax(fitnesses)]] # Best model survives
        mutation_populations = gen_population_mutation(elites, n=int(self.num_populations / 2) - 1, mutation_rate=self.mutation_prob, mutation_power=self.mutation_power)
        crossover_populations = gen_population_crossover(elites, n=int(self.num_populations / 2))
        new_generation.extend(mutation_populations)
        new_generation.extend(crossover_populations)
        random.shuffle(new_generation)
        return new_generation