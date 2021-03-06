# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, optimization, numpy

class Individual(ga.Individual):
    """Represents a solution of a optimization problem."""
    def __init__(self, genotype, fitness_evaluator, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.Individual, self).__init__(genotype, fitness_evaluator)
        if crossover_method == 'one_point':
            self.crossover_method = self.one_point_crossover
        elif crossover_method == 'uniform':
            self.crossover_method = self.uniform_crossover
        else:
            self.crossover_method = crossover_method

        if mutation_method == 'permutation':
            self.mutation_method = self.permutation
        else:
            self.mutation_method = mutation_method

    def mutate(self):
        return self.mutation_method(self)

    def permutation(self, individual):
        """Performs a mutation where two values in the chromosome are swaped."""
        genotype = numpy.array(individual.genotype, copy=True)
        [idx1, idx2] = numpy.random.randint(0, len(genotype), 2)
        aux = individual.genotype[idx1]
        numpy.put(genotype, [idx1], [genotype[idx2]])
        numpy.put(genotype, [idx2], [aux])
        return optimization.Individual(genotype, individual.fitness_evaluator, individual.crossover_method, individual.mutation_method)

    def crossover(self, another_individual):
        return self.crossover_method(another_individual)

    def one_point_crossover(self, another_individual):
        """All data beyond a select index in either individual genotype is swapped between the two parent genotypes."""
        size = len(another_individual.get_genotype())
        genotype1 = numpy.zeros(size, dtype=another_individual.get_genotype().dtype)
        genotype2 = numpy.zeros(size, dtype=another_individual.get_genotype().dtype)
        idx = numpy.random.randint(1, size)
        numpy.put(genotype1, range(0, idx), another_individual.get_genotype()[0:idx])
        numpy.put(genotype1, range(idx, size), self.get_genotype()[idx:size])
        numpy.put(genotype2, range(0, idx), self.get_genotype()[0:idx])
        numpy.put(genotype2, range(idx, size), another_individual.get_genotype()[idx:size])

        return optimization.Individual(genotype1, self.fitness_evaluator, self.crossover_method, self.mutation_method), optimization.Individual(genotype2, self.fitness_evaluator, self.crossover_method, self.mutation_method)

    def uniform_crossover(self, another_individual):
        """A mask defines from which parent genotype data must be copied."""
        size = len(another_individual.get_genotype())
        genotype1 = numpy.zeros(size, dtype=another_individual.get_genotype().dtype)
        genotype2 = numpy.zeros(size, dtype=another_individual.get_genotype().dtype)
        mask = numpy.random.choice([True,False], size=size)
        not_mask = numpy.logical_not(mask)
        genotype1[mask] = self.get_genotype()[mask]
        genotype1[not_mask] = another_individual.get_genotype()[not_mask]
        genotype2[mask] = another_individual.get_genotype()[mask]
        genotype2[not_mask] = self.get_genotype()[not_mask]

        return optimization.Individual(genotype1, self.fitness_evaluator, self.uniform_crossover, self.mutation_method), optimization.Individual(genotype2, self.fitness_evaluator, self.uniform_crossover, self.mutation_method)

ga.Individual.register(Individual)