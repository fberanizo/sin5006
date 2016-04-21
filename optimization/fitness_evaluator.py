# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, optimization, numpy

class FitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on Rastrigin function value."""
        A = 10
        fitness =  A * len(individual.get_genotype())
        for x in individual.get_genotype():
            fitness += numpy.square(x) - (A * numpy.cos(2 * numpy.pi * x))
        return -fitness

ga.FitnessEvaluator.register(FitnessEvaluator)