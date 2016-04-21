# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, optimization, numpy

class Individual(ga.Individual):
    def __init__(self, genotype, fitness_evaluator, crossover_method='one_point'):
        super(optimization.Individual, self).__init__(genotype, fitness_evaluator)
        if crossover_method == 'uniform':
            self.crossover_method = self.uniform_crossover
        else:
            self.crossover_method = self.one_point_crossover

    def mutate(self):
        idx = numpy.random.randint(0, 2)
        value = numpy.random.uniform(low=-5.0, high=5.0)
        numpy.put(self.genotype, [idx], [value])
        self.fitness = self.fitness_evaluator.evaluate(self)

    def crossover(self, another_individual):
        return self.crossover_method(another_individual)

    def one_point_crossover(self, another_individual):
        size = len(another_individual.get_genotype())
        genotype1 = numpy.zeros(size)
        genotype2 = numpy.zeros(size)
        idx = numpy.random.randint(1, size)
        numpy.put(genotype1, range(0, idx), another_individual.get_genotype()[0:idx])
        numpy.put(genotype1, range(idx, size), self.get_genotype()[idx:size])
        numpy.put(genotype2, range(0, idx), self.get_genotype()[0:idx])
        numpy.put(genotype2, range(idx, size), another_individual.get_genotype()[idx:size])

        return optimization.Individual(genotype1, self.fitness_evaluator, self.crossover_method), optimization.Individual(genotype2, self.fitness_evaluator, self.crossover_method)

    def uniform_crossover(self, another_individual):
        size = len(another_individual.get_genotype())
        genotype1 = numpy.zeros(size)
        genotype2 = numpy.zeros(size)
        mask = numpy.random.choice([True,False], size=size)
        not_mask = numpy.logical_not(mask)
        genotype1[mask] = self.get_genotype()[mask]
        genotype1[not_mask] = another_individual.get_genotype()[not_mask]
        genotype2[mask] = another_individual.get_genotype()[mask]
        genotype2[not_mask] = self.get_genotype()[not_mask]

        return optimization.Individual(genotype1, self.fitness_evaluator, self.uniform_crossover), optimization.Individual(genotype2, self.fitness_evaluator, self.uniform_crossover)

ga.Individual.register(Individual)