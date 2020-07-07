import torch
import numpy as np

from torch import Tensor
from typing import List
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.chromosomes import *
from utilities.random_generator import RandomGenerator, NPRandomGenerator

random_generator = NPRandomGenerator()

def select_elites(populations: List[NetworkGenotype], fitnesses, n) -> List[NetworkGenotype]:
    elite_idx = np.argsort(fitnesses)[-n:]
    elites = [populations[i] for i in elite_idx]
    return elites

def mutate(genotype: NetworkGenotype, random_generator: RandomGenerator, mutation_rate) -> NetworkGenotype:
    child = genotype.clone()
    gene_length = len(genotype.genes)
    idx_to_mutate = random_generator.randint(gene_length, int(mutation_rate * gene_length))
    for i in idx_to_mutate: child.genes[i] += torch.randn_like(child.genes[i])
    return child

def crossover(a: NetworkGenotype, b:NetworkGenotype) -> NetworkGenotype:
    c = a.clone()
    for name in a.chromosomes.keys():
        a_chromosome, b_chromosome = a.chromosomes[name], b.chromosomes[name]
        c.chromosomes[name] = a_chromosome.clone() if np.random.rand() <= 0.5 else b_chromosome.clone()
    return c

def gen_population_mutation(parents: List[NetworkGenotype], n, mutation_rate=0.01):
    new_generation = []
    K = np.random.randint(0, len(parents), n)
    parents = [parents[k] for k in K]
    for p in parents: new_generation.append(mutate(p, random_generator, mutation_rate))
    return new_generation

def gen_population_crossover(parents: List[NetworkGenotype], n):
    assert len(parents) >= 2, "Number of parents must be more than 2"
    new_generation = []
    num_parents = len(parents)
    for i in range(n):
        a, b = np.random.randint(0, num_parents, 2)
        while a == b: a, b = np.random.randint(0, num_parents, 2)
        new_generation.append(crossover(parents[a], parents[b]))
    return new_generation