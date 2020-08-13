import torch
import numpy as np

from torch import Tensor
from typing import List
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.network_schema import *
from utilities.random_generator import RandomGenerator, NPRandomGenerator

random_generator = NPRandomGenerator()

def select_elites(populations: List[NetworkGenotype], fitnesses, n) -> List[NetworkGenotype]:
    elite_idx = np.argsort(fitnesses)[-n:]
    elites = [populations[i] for i in elite_idx]
    return elites

def mutate(genotype: NetworkGenotype, random_generator: RandomGenerator, mutation_rate, mutation_power=1.) -> NetworkGenotype:
    child = genotype.clone()
    gene_length = len(genotype.genes)
    idx_to_mutate = random_generator.randint(gene_length, int(mutation_rate * gene_length))
    for i in idx_to_mutate: child.genes[i] += mutation_power * torch.randn_like(child.genes[i])
    return child

def crossover(a: NetworkGenotype, b:NetworkGenotype, random_generator: RandomGenerator) -> NetworkGenotype:
    c1, c2 = a.clone(), b.clone()
    gene_length = len(a.genes)
    idx_to_cross = random_generator.randint(gene_length, int(0.5 * gene_length)) # Uniform crossover w. 50% of genes from each parent
    for i in idx_to_cross: 
        c1.genes[i] = b.genes[i]
        c2.genes[i] = a.genes[i]
    return (c1, c2)

def gen_population_mutation(parents: List[NetworkGenotype], n, mutation_rate=0.01, mutation_power=1.):
    new_generation = []
    K = np.random.randint(0, len(parents), n)
    parents = [parents[k] for k in K]
    for p in parents: new_generation.append(mutate(p, random_generator, mutation_rate, mutation_power))
    return new_generation

def gen_population_crossover(parents: List[NetworkGenotype], n):
    assert len(parents) >= 2, "Number of parents must be more than 2"
    new_generation = []
    num_parents = len(parents)
    for i in range(int(n / 2) + 1):
        a, b = np.random.randint(0, num_parents, 2)
        while a == b: a, b = np.random.randint(0, num_parents, 2)
        c1, c2 = crossover(parents[a], parents[b], random_generator)
        new_generation.append(c1)
        new_generation.append(c2)
    return new_generation[0:n]