# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, optimization, numpy

class IndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point'):
        super(optimization.IndividualFactory, self).__init__()
        self.crossover_method = crossover_method

    def create(self):
        """Creates individuals which [x,y] values are uniformly distributed over -5.0 and 5.0."""
        genotype = numpy.random.uniform(low=-5.0, high=5.0, size=2)
        fitness_evaluator = optimization.FitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method)

ga.IndividualFactory.register(IndividualFactory)