import numpy as np
import random

from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm.opetators import *
from utilities.pareto_solution import find_pareto
from utilities.ga_helpers import calculate_fitnesses
from utilities.learning_info import LearningInfo

class DiversityPromotedGA(GeneticAlgorithm):
    def __init__(self, num_populations, fitness_evaluator, diversity_evaluator, mutation_prob, mutation_power):
        self.num_populations = num_populations
        self.fitness_evaluator = fitness_evaluator
        self.diversity_evaluator = diversity_evaluator
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        
    def run(self, populations, num_generations, max_iterations=None, num_episodes_eval=1, visualize=False, num_workers=None, run_mode=None):
        avg_fitnesses, max_fitnesses = [], []
        solution = None
        for gen in range(num_generations):
            fitnesses, final_states = calculate_fitnesses(populations, self.fitness_evaluator, gen, num_workers, run_mode, max_iterations, num_episodes_eval, visualize)
            avg_fitnesses.append(fitnesses.mean())
            max_fitnesses.append(fitnesses.max())
            solution = populations[fitnesses.argmax()]
            new_gen = self.new_generation(populations, fitnesses, final_states)
            populations = new_gen
        
        return solution, LearningInfo(avg_fitnesses, max_fitnesses)
    
    def new_generation(self, old_gen, fitnesses, final_states):
        diversity_scores = self.diversity_evaluator.eval_diversity(final_states)
        non_dominated_idx = find_pareto([fitnesses, diversity_scores])
        elites = [old_gen[i] for i in non_dominated_idx]
        pos = [final_states[i] for i in non_dominated_idx]
        scores = [diversity_scores[i] for i in non_dominated_idx]
        new_generation = []
        new_generation.extend(elites) # Elites model survives
        mutation_populations = gen_population_mutation(elites, n=self.num_populations - len(elites), mutation_rate=self.mutation_prob, mutation_power=self.mutation_power)
        new_generation.extend(mutation_populations)
        random.shuffle(new_generation)
        return new_generation