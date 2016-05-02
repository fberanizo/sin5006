# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, optimization, numpy, struct

def binary(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

class RastriginFloatIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.RastriginFloatIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x,y] values are uniformly distributed over -5.0 and 5.0."""
        genotype = numpy.random.uniform(low=-5.0, high=5.0, size=2)
        fitness_evaluator = optimization.RastriginFloatFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        """Performs a basic mutation where one value in the chromosome is replaced by another valid value."""
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.uniform(low=-5.0, high=5.0)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(RastriginFloatIndividualFactory)

class RastriginBinaryIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.RastriginBinaryIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x,y] values are represented by 32 bits."""
        genotype = map(binary, numpy.random.uniform(low=-5.0, high=5.0, size=2))
        genotype = numpy.array(list("".join(genotype)), dtype=int)
        fitness_evaluator = optimization.RastriginBinaryFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        """Performs a basic mutation where one value in the chromosome is replaced by another valid value."""
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.randint(2)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(RastriginBinaryIndividualFactory)

class XSquareFloatIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.XSquareFloatIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x1,x2,...,x30] values are uniformly distributed over -100.0 and 100.0."""
        genotype = numpy.random.uniform(low=-100.0, high=100.0, size=30)
        fitness_evaluator = optimization.XSquareFloatFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        """Performs a basic mutation where one value in the chromosome is replaced by another valid value."""
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.uniform(low=-100.0, high=100.0)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(XSquareFloatIndividualFactory)

class XSquareBinaryIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.XSquareBinaryIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x,y] values are represented by 32 bits."""
        genotype = map(binary, numpy.random.uniform(low=-100.0, high=100.0, size=30))
        genotype = numpy.array(list("".join(genotype)), dtype=int)
        fitness_evaluator = optimization.XSquareBinaryFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        """Performs a basic mutation where one value in the chromosome is replaced by another valid value."""
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.randint(2)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(XSquareBinaryIndividualFactory)

class XAbsoluteSquareFloatIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.XAbsoluteSquareFloatIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x1,x2,...,x30] values are uniformly distributed over -100.0 and 100.0."""
        genotype = numpy.random.uniform(low=-100.0, high=100.0, size=30)
        fitness_evaluator = optimization.XAbsoluteSquareFloatFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        """Performs a basic mutation where one value in the chromosome is replaced by another valid value."""
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.uniform(low=-100.0, high=100.0)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(XAbsoluteSquareFloatIndividualFactory)

class XAbsoluteSquareBinaryIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.XAbsoluteSquareBinaryIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x,y] values are represented by 32 bits."""
        genotype = map(binary, numpy.random.uniform(low=-100.0, high=100.0, size=30))
        genotype = numpy.array(list("".join(genotype)), dtype=int)
        fitness_evaluator = optimization.XAbsoluteSquareBinaryFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        """Performs a basic mutation where one value in the chromosome is replaced by another valid value."""
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.randint(2)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(XAbsoluteSquareBinaryIndividualFactory)

class SineXSquareRootFloatIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.SineXSquareRootFloatIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x1,x2,...,x30] values are uniformly distributed over -500.0 and 500.0."""
        genotype = numpy.random.uniform(low=-500.0, high=500.0, size=30)
        fitness_evaluator = optimization.SineXSquareRootFloatFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        """Performs a basic mutation where one value in the chromosome is replaced by another valid value."""
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.uniform(low=-500.0, high=500.0)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(SineXSquareRootFloatIndividualFactory)

class SineXSquareRootBinaryIndividualFactory(ga.IndividualFactory):
    def __init__(self, crossover_method='one_point', mutation_method='permutation'):
        super(optimization.SineXSquareRootBinaryIndividualFactory, self).__init__()
        self.crossover_method = crossover_method
        
        if mutation_method == 'basic_mutation':
            self.mutation_method = self.basic_mutation
        else:
            self.mutation_method = mutation_method

    def create(self):
        """Creates individuals which [x,y] values are represented by 32 bits."""
        genotype = map(binary, numpy.random.uniform(low=-500.0, high=500.0, size=30))
        genotype = numpy.array(list("".join(genotype)), dtype=int)
        fitness_evaluator = optimization.SineXSquareRootBinaryFitnessEvaluator()
        return optimization.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

    def basic_mutation(self_individual, individual):
        """Performs a basic mutation where one value in the chromosome is replaced by another valid value."""
        idx = numpy.random.randint(0, len(individual.genotype))
        value = numpy.random.randint(2)
        numpy.put(individual.genotype, [idx], [value])
        individual.fitness = individual.fitness_evaluator.evaluate(individual)
        return individual

ga.IndividualFactory.register(SineXSquareRootBinaryIndividualFactory)