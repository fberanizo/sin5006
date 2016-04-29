# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, optimization, numpy

class RastriginFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on Rastrigin function value."""
        A = 10
        fitness =  A * len(individual.get_genotype())
        for x in individual.get_genotype():
            fitness += numpy.square(x) - (A * numpy.cos(2 * numpy.pi * x))
        return -fitness

ga.FitnessEvaluator.register(RastriginFitnessEvaluator)

class XSquareFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum xi², i=1 to 30 function value."""
        D = 30
        fitness =  D * len(individual.get_genotype())
        for x in individual.get_genotype():
            fitness += numpy.square(x)
        return -fitness

ga.FitnessEvaluator.register(XSquareFitnessEvaluator)

class XAbsoluteSquareFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum |xi + 0.5|², i=1 to 30 function value."""
        D = 30
        fitness =  D * len(individual.get_genotype())
        for x in individual.get_genotype():
            fitness += numpy.square(numpy.absolute(x + 0.5))
        return -fitness

ga.FitnessEvaluator.register(XAbsoluteSquareFitnessEvaluator)

class SineXSquareRootFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum -xi*sin(sqrt(|xi|)), i=1 to 30 function value."""
        D = 30
        fitness =  D * len(individual.get_genotype())
        for x in individual.get_genotype():
            fitness += numpy.negative(x)*numpy.sin(numpy.sqrt(numpy.absolute(x)))
        return -fitness

ga.FitnessEvaluator.register(SineXSquareRootFitnessEvaluator)