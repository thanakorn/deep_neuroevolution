import random

from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm.opetators import crossover, mutate
from utilities.random_generator import NPRandomGenerator
from utilities.ga_helpers import calculate_fitnesses
from utilities import random_generator

class DeterministicCrowdingGA(CrowdingGA):    
    def run(self, populations, num_generations, max_iterations=None, num_episodes_eval=1, visualize=False, num_workers=None, run_mode=None):
        solution = None
        for gen in range(num_generations):
            fitnesses = calculate_fitnesses(populations, self.fitness_evaluator, gen, num_workers, run_mode, max_iterations, num_episodes_eval, visualize)
            solution = populations[fitnesses.argmax()]
            new_gen = self.new_generation(populations, fitnesses, gen, num_workers, run_mode, max_iterations, num_episodes_eval, visualize)
            populations = new_gen
        return solution
    
    def new_generation(self, old_gen, fitnesses, *args):
        offsprings = []
        for p1, p2 in zip(old_gen[0::2], old_gen[1::2]):
            c1, c2 = crossover(p1, p2, random_generator)
            offsprings.append(mutate(c1, random_generator, self.mutation_prob, self.mutation_power))
            offsprings.append(mutate(c2, random_generator, self.mutation_prob, self.mutation_power))
        offspring_fitnesses = calculate_fitnesses(offsprings, self.fitness_evaluator, args)
        
        new_generation = []
        for i in range(self.num_populations):
            new_pop = old_gen[i] if fitnesses[i] > offspring_fitnesses[i] else offsprings[i] # Offspring i is a child of population i
            new_generation.append(new_pop)
        
        random.shuffle(new_generation)
        return new_generation
    
