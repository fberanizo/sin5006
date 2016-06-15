# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, vrp, numpy, struct, math

class BOCIndividual(ga.Individual):
    """Represents a solution of a vehicle routing problem."""
    def __init__(self, genotype, fitness_evaluator):
        super(vrp.Individual, self).__init__(genotype, fitness_evaluator)

    def mutate(self):
        return self.mutation_method(self)

    def permutation(self, individual):
        """Performs a mutation where two values in the chromosome are swaped."""
        genotype = numpy.array(individual.genotype, copy=True)
        [idx1, idx2] = numpy.random.randint(0, len(genotype), 2)
        aux = individual.genotype[idx1]
        numpy.put(genotype, [idx1], [genotype[idx2]])
        numpy.put(genotype, [idx2], [aux])
        return vrp.BOCIndividual(genotype, individual.fitness_evaluator)

    def crossover(self, another_individual):
        """Performs Biggest Overlap Crossover"""
        pass
