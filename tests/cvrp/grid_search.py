# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy, itertools, operator, pandas

class GridSearch(object):
    def __init__(self, genetic_algorithm, params, repeat=2):
        self.genetic_algorithm = genetic_algorithm
        self.params = (dict(zip(params, x)) for x in itertools.product(*params.itervalues()))
        self.grid_scores = pandas.DataFrame([], columns=["params", "best_fitness"])
        self.size = reduce(operator.mul, map(len, params.values()))
        self.repeat = repeat

    def search(self, optimal):
        """Performs a grid search, varying parameters and obtaining fitnesses.
           The GA is repeated for the same parametrization the number of times informed."""
        
        for idx, params in enumerate(self.params):
            bar_length = 20
            percent = float(idx) / self.size
            hashes = '#' * int(round(percent * bar_length))
            spaces = ' ' * (bar_length - len(hashes))
            sys.stdout.write("\rPerforming grid search: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
            sys.stdout.flush()

            best_fitnesses = []

            for x in xrange(self.repeat):
                self.genetic_algorithm.set_params(**params)
                self.genetic_algorithm.init_population()
                self.genetic_algorithm.evolve()

                individual = self.genetic_algorithm.result()
                genotype = individual.get_genotype()
                best_individual = self.genetic_algorithm.result()
                best_fitness = best_individual.get_fitness()
                best_fitnesses.append(best_fitness)

            info_mean = pandas.DataFrame([[params, numpy.mean(best_fitness)]], columns=["params", "best_fitness"])
            self.grid_scores = self.grid_scores.append(info_mean, ignore_index=True)

        sys.stdout.write("\rPerforming grid search: [{0}] {1}%\n\n".format(hashes + spaces, int(round(100))))
        sys.stdout.flush()

    def get_grid_scores(self):
        """Returns a list of tuples that have the attributes:
                params: the parameters used by the GA
                best_fitness: the mean of best fitnesses of each iteration
        """
        return self.grid_scores

    def get_best_parametrization(self):
        """Returns a list of tuples that have the attributes:
                params: the parameters used by the GA
                best_fitness: the mean of best fitnesses of each iteration
        """
        return max(self.grid_scores, key=lambda x: x[1])

