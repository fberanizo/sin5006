# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import unittest, grid_search, utils, ga, optimization, time, numpy, itertools, pandas

class Rastrigin(unittest.TestCase):
    """Test cases for Ratrigin function. 
       Elitism, binary and float array chromossomes, different types of crossover and mutation are tested.
    """
    def test_case1(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: one-point crossover
        Mutation operator: basic (replaces x or y by another valid value)
        Elitism is disabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: [8, 19, 47, 115, 282, 689, 1680, 4096]
                (numbers spaced evenly on a log scale)
            crossover rate, reproduction rate, mutation rate: varying from 0.0 to 1.0
                (all possible combinations that sum to 1.0 ex: [0.2, 0.5, 0.3])
        """
        sys.stdout.write("Starting test_case1: ONE-POINT CROSSOVER, BASIC MUTATION, ELITISM DISABLED\n")

        params = {
            "population_size": numpy.logspace(3, 12, base=2, num=8, dtype=int),
            "operators_rate": filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.0, 1.1, .1), repeat=3)),
            "elitism": [False],
            "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
        }
        solver = ga.GeneticAlgorithm(optimization.RastriginFloatIndividualFactory(crossover_method='one_point', mutation_method='basic_mutation'))
        grid = grid_search.GridSearch(solver, params)
        grid.search(0.0)
        grid_scores = grid.get_grid_scores()

        filepath ='/home/fabio/sin5006/tests/results/rastrigin/test_case1.csv'
        utils.save_scores(filepath, grid_scores)
        dataset = pandas.pivot_table(pandas.read_csv(filepath), index='Operators', columns='Population', values='Fitness', aggfunc=numpy.sum)
        utils.plot_heatmap('/home/fabio/sin5006/tests/results/rastrigin/test_case1_hm.png', dataset)

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/rastrigin/test_case1.csv\n")
        assert True

    def test_case2(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: one-point crossover
        Mutation operator: basic (replaces x or y by another valid value)
        Elitism is enabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: [8, 19, 47, 115, 282, 689, 1680, 4096]
                (numbers spaced evenly on a log scale)
            crossover rate, reproduction rate, mutation rate: varying from 0.0 to 1.0
                (all possible combinations that sum to 1.0 ex: [0.2, 0.5, 0.3])
        """
        sys.stdout.write("Starting test_case2: ONE-POINT CROSSOVER, BASIC MUTATION, ELITISM ENABLED\n")

        params = {
            "population_size": numpy.logspace(3, 12, base=2, num=8, dtype=int),
            "operators_rate": filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.0, 1.1, .1), repeat=3)),
            "elitism": [True],
            "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
        }
        solver = ga.GeneticAlgorithm(optimization.RastriginFloatIndividualFactory(crossover_method='one_point', mutation_method='basic_mutation'))
        grid = grid_search.GridSearch(solver, params)
        grid.search(0.0)
        grid_scores = grid.get_grid_scores()

        filepath ='/home/fabio/sin5006/tests/results/rastrigin/test_case2.csv'
        utils.save_scores(filepath, grid_scores)
        dataset = pandas.pivot_table(pandas.read_csv(filepath), index='Operators', columns='Population', values='Fitness', aggfunc=numpy.sum)
        utils.plot_heatmap('/home/fabio/sin5006/tests/results/rastrigin/test_case2_hm.png', dataset)

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/rastrigin/test_case2.csv\n")
        assert True

    def test_case3(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: uniform crossover
        Mutation operator: basic (replaces x or y by another valid value)
        Elitism is disabled
        Termination criteria: number of generations = 100

        Parameters:
            population_size: [8, 19, 47, 115, 282, 689, 1680, 4096]
                (numbers spaced evenly on a log scale)
            crossover rate, reproduction rate, mutation rate: varying from 0.0 to 1.0
                (all possible combinations that sum to 1.0 ex: [0.2, 0.5, 0.3])
        """
        sys.stdout.write("Starting test_case3: UNIFORM CROSSOVER, BASIC MUTATION, ELITISM DISABLED\n")

        params = {
            "population_size": numpy.logspace(3, 12, base=2, num=8, dtype=int),
            "operators_rate": filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.0, 1.1, .1), repeat=3)),
            "elitism": [False],
            "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
        }
        solver = ga.GeneticAlgorithm(optimization.RastriginFloatIndividualFactory(crossover_method='uniform', mutation_method='basic_mutation'))
        grid = grid_search.GridSearch(solver, params)
        grid.search(0.0)
        grid_scores = grid.get_grid_scores()

        filepath = '/home/fabio/sin5006/tests/results/rastrigin/test_case3.csv'
        utils.save_scores(filepath, grid_scores)
        dataset = pandas.pivot_table(pandas.read_csv(filepath), index='Operators', columns='Population', values='Fitness', aggfunc=numpy.sum)
        utils.plot_heatmap('/home/fabio/sin5006/tests/results/rastrigin/test_case3_hm.png', dataset)

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/rastrigin/test_case3.csv\n")
        assert True

    def test_case4(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: one-point crossover
        Mutation operator: permutation
        Elitism is disabled
        Termination criteria: number of generations = 100

        Parameters:
            population_size: [8, 19, 47, 115, 282, 689, 1680, 4096]
                (numbers spaced evenly on a log scale)
            crossover rate, reproduction rate, mutation rate: varying from 0.0 to 1.0
                (all possible combinations that sum to 1.0 ex: [0.2, 0.5, 0.3])
        """
        sys.stdout.write("Starting test_case4: ONE-POINT CROSSOVER, PERMUTATION MUTATION, ELITISM DISABLED\n")

        params = {
            "population_size": numpy.logspace(3, 12, base=2, num=8, dtype=int),
            "operators_rate": filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.0, 1.1, .1), repeat=3)),
            "elitism": [False],
            "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
        }
        solver = ga.GeneticAlgorithm(optimization.RastriginFloatIndividualFactory(crossover_method='one_point', mutation_method='permutation'))
        grid = grid_search.GridSearch(solver, params)
        grid.search(0.0)
        grid_scores = grid.get_grid_scores()

        filepath = '/home/fabio/sin5006/tests/results/rastrigin/test_case4.csv'
        utils.save_scores(filepath, grid_scores)
        dataset = pandas.pivot_table(pandas.read_csv(filepath), index='Operators', columns='Population', values='Fitness', aggfunc=numpy.sum)
        utils.plot_heatmap('/home/fabio/sin5006/tests/results/rastrigin/test_case4_hm.png', dataset)

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/rastrigin/test_case4.csv\n")
        assert True

    def test_case5(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: one-point crossover
        Mutation operator: basic (replaces x or y by another valid value)
        Elitism is disabled
        Termination criteria: execution_time = 2s

        Parameters:
            population_size: [8, 19, 47, 115, 282, 689, 1680, 4096]
                (numbers spaced evenly on a log scale)
            crossover rate, reproduction rate, mutation rate: varying from 0.0 to 1.0
                (all possible combinations that sum to 1.0 ex: [0.2, 0.5, 0.3])
        """
        sys.stdout.write("Starting test_case5: ONE-POINT CROSSOVER, BASIC MUTATION, ELITISM DISABLED, 2 SECONDS OF EXECUTION TIME\n")

        params = {
            "population_size": numpy.logspace(3, 12, base=2, num=8, dtype=int),
            "operators_rate": filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.0, 1.1, .1), repeat=3)),
            "elitism": [False],
            "termination_criteria": [ ga.ExecutionTimeTerminationCriteria(time_in_seconds=2.0) ]
        }
        solver = ga.GeneticAlgorithm(optimization.RastriginFloatIndividualFactory(crossover_method='one_point', mutation_method='basic_mutation'))
        grid = grid_search.GridSearch(solver, params)
        grid.search(0.0)
        grid_scores = grid.get_grid_scores()

        filepath = '/home/fabio/sin5006/tests/results/rastrigin/test_case5.csv'
        utils.save_scores(filepath, grid_scores)
        dataset = pandas.pivot_table(pandas.read_csv(filepath), index='Operators', columns='Population', values='Fitness', aggfunc=numpy.sum)
        utils.plot_heatmap('/home/fabio/sin5006/tests/results/rastrigin/test_case5_hm.png', dataset)

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/rastrigin/test_case5.csv\n")
        assert True

    def test_case6(self):
        """
        Chromosome: binary array containing x and y values
        Selection: roulette-wheel
        Crossover operator: one-point crossover
        Mutation operator: basic (replaces x or y by another valid value)
        Elitism is disabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: [8, 19, 47, 115, 282, 689, 1680, 4096]
                (numbers spaced evenly on a log scale)
            crossover rate, reproduction rate, mutation rate: varying from 0.0 to 1.0
                (all possible combinations that sum to 1.0 ex: [0.2, 0.5, 0.3])
        """
        sys.stdout.write("Starting test_case6: BINARY CHROMOSOME, ONE-POINT CROSSOVER, BASIC MUTATION, ELITISM DISABLED\n")

        params = {
            "population_size": numpy.logspace(3, 12, base=2, num=8, dtype=int),
            "operators_rate": filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.0, 1.1, .1), repeat=3)),
            "elitism": [False],
            "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
        }
        solver = ga.GeneticAlgorithm(optimization.RastriginBinaryIndividualFactory(crossover_method='one_point', mutation_method='basic_mutation'))
        grid = grid_search.GridSearch(solver, params)
        grid.search(0.0)
        grid_scores = grid.get_grid_scores()

        filepath ='/home/fabio/sin5006/tests/results/rastrigin/test_case6.csv'
        utils.save_scores(filepath, grid_scores)
        dataset = pandas.pivot_table(pandas.read_csv(filepath), index='Operators', columns='Population', values='Fitness', aggfunc=numpy.sum)
        utils.plot_heatmap('/home/fabio/sin5006/tests/results/rastrigin/test_case6_hm.png', dataset)

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/rastrigin/test_case6.csv\n")
        assert True

if __name__ == '__main__':
    unittest.main()
