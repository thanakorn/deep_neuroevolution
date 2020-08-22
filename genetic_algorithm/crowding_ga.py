import random

from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm.opetators import crossover, mutate
from utilities.random_generator import NPRandomGenerator
from utilities.ga_helpers import calculate_fitnesses
from utilities.learning_info import LearningInfo
from genetic_algorithm.opetators import random_generator

class DeterministicCrowdingGA(GeneticAlgorithm):    
    def run(self, populations, num_generations, max_iterations=None, num_episodes_eval=1, visualize=False, num_workers=None, run_mode=None):
        avg_fitnesses, max_fitnesses = [], []
        solution = None
        for gen in range(num_generations):
            fitnesses = calculate_fitnesses(populations, self.fitness_evaluator, gen, num_workers, run_mode, max_iterations, num_episodes_eval, visualize)
            avg_fitnesses.append(fitnesses.mean())
            max_fitnesses.append(fitnesses.max())
            solution = populations[fitnesses.argmax()]
            new_gen = self.new_generation(populations, fitnesses, gen, num_workers, run_mode, max_iterations, num_episodes_eval, visualize)
            populations = new_gen
        return solution, LearningInfo(avg_fitnesses, max_fitnesses)
    
    def gen_offsprings(self, parents):
        offsprings = []
        for p1, p2 in zip(parents[0::2], parents[1::2]):
            c1, c2 = crossover(p1, p2, random_generator)
            offsprings.append(mutate(c1, random_generator, self.mutation_prob, self.mutation_power))
            offsprings.append(mutate(c2, random_generator, self.mutation_prob, self.mutation_power))
        return offsprings
    
    def replace_parents(self, parents, parent_fitnesses, offsprings, offspring_fitnesses):
        new_generation = []
        for i in range(self.num_populations):
            p = parents[i] if parent_fitnesses[i] > offspring_fitnesses[i] else offsprings[i] # Offspring i is a child of population i
            new_generation.append(p)
        return new_generation
    
    def new_generation(self, old_gen, fitnesses, *args):
        gen, num_workers, run_mode, max_iterations, num_episodes_eval, visualize = args
        offsprings = self.gen_offsprings(old_gen)
        offspring_fitnesses = calculate_fitnesses(offsprings, self.fitness_evaluator, gen, num_workers, run_mode, max_iterations, num_episodes_eval, visualize)
        new_generation = self.replace_parents(old_gen, fitnesses, offsprings, offspring_fitnesses)
        random.shuffle(new_generation)
        return new_generation
    
