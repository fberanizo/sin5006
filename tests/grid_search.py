# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy, itertools, operator

class GridSearch(object):
    def __init__(self, genetic_algorithm, params, repeat=10):
        self.genetic_algorithm = genetic_algorithm
        self.params = (dict(zip(params, x)) for x in itertools.product(*params.itervalues()))
        self.repeat = repeat
        self.grid_scores = []
        self.size = reduce(operator.mul, map(len, params.values()))

    def search(self, optimal):
        """Performs a grid search, varying parameters and obtaining fitnesses.
           The GA is repeated for the same parametrization the number of times informed."""
        for idx, params in enumerate(self.params):
            print "Testing parameters " + str(idx) + " of " + str(self.size) + "."
            self.genetic_algorithm.set_params(**params)
            self.genetic_algorithm.init_population()
            self.genetic_algorithm.evolve()

            best_fitnesses = []
            for x in xrange(self.repeat):
                individual = self.genetic_algorithm.result()
                genotype = individual.get_genotype()
                best_individual = self.genetic_algorithm.result()
                best_fitnesses.append(best_individual.get_fitness())

            mean_best_fitness = numpy.mean(best_fitnesses)
            self.grid_scores.append({"params": params, "mean_best_fitness": mean_best_fitness})

    def get_grid_scores(self):
        """Returns a list of tuples that have the attributes:
                params: the parameters used by the GA
                mean_best_fitness: the mean of best fitnesses of each iteration
        """
        return self.grid_scores

    def get_best_parametrization(self):
        """Returns a list of tuples that have the attributes:
                params: the parameters used by the GA
                mean_best_fitness: the mean of best fitnesses of each iteration
        """
        return max(self.grid_scores, key=lambda x: x[1])

