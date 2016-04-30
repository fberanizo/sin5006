# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, optimization, numpy

class RastriginIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.RastriginIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x,y] values are uniformly distributed over -5.0 and 5.0."""
        genotype = numpy.random.uniform(low=-5.0, high=5.0, size=2)
        fitness_evaluator = optimization.RastriginFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.uniform(low=-5.0, high=5.0)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(RastriginIndividualFactory)

class XSquareIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='default'):
        super(optimization.XSquareIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x1,x2,...,x30] values are uniformly distributed over -100.0 and 100.0."""
        genotype = numpy.random.uniform(low=-100.0, high=100.0, size=30)
        fitness_evaluator = optimization.XSquareFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

ga.IndividualFactory.register(XSquareIndividualFactory)

class XAbsoluteSquareIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='default'):
        super(optimization.XAbsoluteSquareIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x1,x2,...,x30] values are uniformly distributed over -100.0 and 100.0."""
        genotype = numpy.random.uniform(low=-100.0, high=100.0, size=30)
        fitness_evaluator = optimization.XAbsoluteSquareFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

ga.IndividualFactory.register(XAbsoluteSquareIndividualFactory)

class SineXSquareRootIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='default'):
        super(optimization.SineXSquareRootIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x1,x2,...,x30] values are uniformly distributed over -500.0 and 500.0."""
        genotype = numpy.random.uniform(low=-500.0, high=500.0, size=30)
        fitness_evaluator = optimization.SineXSquareRootFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

ga.IndividualFactory.register(SineXSquareRootIndividualFactory)