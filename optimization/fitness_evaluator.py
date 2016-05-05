# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, optimization, numpy, struct, math

def as_float32(s):
    return struct.unpack("f", struct.pack("I", bits2int(s)))

def bits2int(bits):
    bits = [int(x) for x in bits[::-1]]
    x = 0
    for i in range(len(bits)):
        x += bits[i]*2**i
    return x

class RastriginFloatFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on Rastrigin function value."""
        A = 10
        fitness = A * len(individual.get_genotype())
        for x in individual.get_genotype():
            fitness += numpy.square(x) - (A * numpy.cos(2 * numpy.pi * x))
        return -fitness

ga.FitnessEvaluator.register(RastriginFloatFitnessEvaluator)

class RastriginBinaryFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on Rastrigin function value."""
        A = 10
        fitness = A * (len(individual.get_genotype())/32)
        for idx in xrange(0, len(individual.get_genotype()), 32):
            x = as_float32("".join(map(str, individual.get_genotype()[idx:idx+32])))[0]
            print "============="
            print individual.get_genotype()[idx:idx+32]
            print "".join(map(str, individual.get_genotype()[idx:idx+32]))
            print str(bits2int("".join(map(str, individual.get_genotype()[idx:idx+32])))[0])
            print "============="
            fitness += numpy.square(x) - (A * numpy.cos(2 * numpy.pi * x))

        if math.isnan(fitness):
            print individual.get_genotype()[idx:idx+32]
            sys.exit(0)
        return -fitness

ga.FitnessEvaluator.register(RastriginBinaryFitnessEvaluator)

class XSquareFloatFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum xi², i=1 to 30 function value."""
        D = 30
        fitness = 0.0
        for x in individual.get_genotype():
            fitness += numpy.square(x)
        return -fitness

ga.FitnessEvaluator.register(XSquareFloatFitnessEvaluator)

class XSquareBinaryFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum xi², i=1 to 30 function value."""
        D = 30
        fitness = 0.0
        for idx in xrange(0, len(individual.get_genotype()), 32):
            x = as_float32("".join(map(str, individual.get_genotype()[idx:idx+32])))[0]
            fitness += numpy.square(x)
        return -fitness

ga.FitnessEvaluator.register(XSquareBinaryFitnessEvaluator)

class XAbsoluteSquareFloatFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum |xi + 0.5|², i=1 to 30 function value."""
        D = 30
        fitness = 0.0
        for x in individual.get_genotype():
            fitness += numpy.square(numpy.absolute(x + 0.5))
        return -fitness

ga.FitnessEvaluator.register(XAbsoluteSquareFloatFitnessEvaluator)

class XAbsoluteSquareBinaryFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum |xi + 0.5|², i=1 to 30 function value."""
        D = 30
        fitness = 0.0
        for idx in xrange(0, len(individual.get_genotype()), 32):
            x = as_float32("".join(map(str, individual.get_genotype()[idx:idx+32])))[0]
            fitness += numpy.square(numpy.absolute(x + 0.5))
        return -fitness

ga.FitnessEvaluator.register(XAbsoluteSquareBinaryFitnessEvaluator)

class SineXSquareRootFloatFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum -xi*sin(sqrt(|xi|)), i=1 to 30 function value."""
        D = 30
        fitness = 0.0
        for x in individual.get_genotype():
            fitness += numpy.negative(x)*numpy.sin(numpy.sqrt(numpy.absolute(x)))
        return -fitness

ga.FitnessEvaluator.register(SineXSquareRootFloatFitnessEvaluator)

class SineXSquareRootBinaryFitnessEvaluator(ga.FitnessEvaluator):
    def evaluate(self, individual):
        """Evaluates individual based on sum -xi*sin(sqrt(|xi|)), i=1 to 30 function value."""
        D = 30
        fitness = 0.0
        for idx in xrange(0, len(individual.get_genotype()), 32):
            x = as_float32("".join(map(str, individual.get_genotype()[idx:idx+32])))[0]
            fitness += numpy.negative(x)*numpy.sin(numpy.sqrt(numpy.absolute(x)))
        return -fitness

ga.FitnessEvaluator.register(SineXSquareRootBinaryFitnessEvaluator)