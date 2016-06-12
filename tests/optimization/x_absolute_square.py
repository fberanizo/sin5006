# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import unittest, grid_search, utils, ga, optimization, time, numpy, itertools, pandas, matplotlib

class XAbsoluteSquare(unittest.TestCase):
    """Test cases for sum |xi + 0.5|², i=1 to 30.
       Elitism, binary and float array chromossomes, different types of crossover and mutation are tested.
    """
    # def test_grid_search_100_generations(self):
    #     """
    #     Chromosome: float array containing x1,..,x30 values
    #     Selection: roulette-wheel
    #     Crossover operator: one-point crossover
    #     Mutation operator: basic (replaces xi by another valid value)
    #     Elitism is disabled
    #     Termination criteria: number of generations = 100
    #     Parameters:
    #         population_size: [8, 27, 97, 337, 1176, 4096]
    #             (numbers spaced evenly on a log scale)
    #         crossover rate, reproduction rate, mutation rate: varying from 0.125 to 0.875
    #             (all possible combinations that sum to 1.0 varying from [0.125 to 0.875], step = 0.125
    #     """
    #     sys.stdout.write("Starting test_grid_search_100_generations: ONE-POINT CROSSOVER, BASIC MUTATION, ELITISM DISABLED\n")

    #     params = {
    #         "population_size": numpy.logspace(3, 12, base=2, num=6, dtype=int),
    #         "operators_rate": filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.125, 0.875, .125), repeat=3)),
    #         "elitism": [False],
    #         "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
    #     }
    #     solver = ga.GeneticAlgorithm(optimization.XAbsoluteSquareFloatIndividualFactory(crossover_method='one_point', mutation_method='basic_mutation'))
    #     grid = grid_search.GridSearch(solver, params)
    #     grid.search(0.0)
    #     grid_scores = grid.get_grid_scores()

    #     filepath ='/home/fabio/sin5006/tests/results/xabsolutesquare/test_grid_search_100_generations.csv'
    #     utils.save_scores(filepath, grid_scores)

    #     sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/xabsolutesquare/test_grid_search_100_generations.csv\n")
    #     assert True

    # def test_grid_search_1_second(self):
    #     """
    #     Chromosome: float array containing x1,..,x30 values
    #     Selection: roulette-wheel
    #     Crossover operator: one-point crossover
    #     Mutation operator: basic (replaces xi by another valid value)
    #     Elitism is disabled
    #     Termination criteria: execution_time = 1s
    #     Parameters:
    #         population_size: [8, 27, 97, 337, 1176, 4096]
    #             (numbers spaced evenly on a log scale)
    #         crossover rate, reproduction rate, mutation rate: varying from 0.125 to 0.875
    #             (all possible combinations that sum to 1.0 varying from [0.125 to 0.875], step = 0.125
    #     """
    #     sys.stdout.write("Starting test_grid_search_1_second: ONE-POINT CROSSOVER, BASIC MUTATION, ELITISM DISABLED\n")

    #     params = {
    #         "population_size": numpy.logspace(3, 12, base=2, num=6, dtype=int),
    #         "operators_rate": filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.125, 0.875, .125), repeat=3)),
    #         "elitism": [False],
    #         "termination_criteria": [ ga.ExecutionTimeTerminationCriteria(time_in_seconds=1.0) ]
    #     }
    #     solver = ga.GeneticAlgorithm(optimization.XAbsoluteSquareFloatIndividualFactory(crossover_method='one_point', mutation_method='basic_mutation'))
    #     grid = grid_search.GridSearch(solver, params)
    #     grid.search(0.0)
    #     grid_scores = grid.get_grid_scores()

    #     filepath ='/home/fabio/sin5006/tests/results/xabsolutesquare/test_grid_search_1_second.csv'
    #     utils.save_scores(filepath, grid_scores)

    #     sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/xabsolutesquare/test_grid_search_1_second.csv\n")
    #     assert True

    def test_base(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: one-point crossover
        Mutation operator: basic (replaces x or y by another valid value)
        Elitism is disabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: 27
            reproduction rate: 0.125
            crossover rate: 0.75
            mutation rate: 0.125
        """
        sys.stdout.write("Starting test_elitism: ONE-POINT CROSSOVER, BASIC MUTATION, ELITISM ENABLED\n")

        reproduction = 0.125
        crossover = 0.75
        mutation = 0.125
        population_size = 27

        repeat = 1

        info = pandas.DataFrame([], columns=["generation", "mean", "max", "std", "execution"])
        for idx in xrange(repeat):
            individual_factory = optimization.XAbsoluteSquareFloatIndividualFactory(crossover_method='one_point', mutation_method='basic_mutation')
            termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
            solver = ga.GeneticAlgorithm(individual_factory, population_size=population_size, reproduction=reproduction, crossover=crossover, mutation=mutation, elitism=False, termination_criteria=termination_criteria)
            solver.set_params()
            solver.init_population()
            solver.evolve()
            i = solver.get_generation_info()
            i["execution"] = idx
            info = info.append(i)
            
            matplotlib.pyplot.plot(i['generation'], i['max'], "r", label="melhor", linewidth=2)
            matplotlib.pyplot.plot(i['generation'], i['mean'], "b", label="media", linewidth=2)
            matplotlib.pyplot.plot(i['generation'], i['std'], "k.", label="desvio")

        info.to_csv('/home/fabio/sin5006/tests/results/xabsolutesquare/test_base.csv', sep=',', index=False)

        legend = matplotlib.pyplot.legend(loc='lower right', numpoints=1)
        matplotlib.pyplot.xlabel("geracoes")
        matplotlib.pyplot.ylabel("fitness")
        matplotlib.pyplot.show()

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/xabsolutesquare/test_base.csv\n")
        assert True

    def test_elitism(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: one-point crossover
        Mutation operator: basic (replaces x or y by another valid value)
        Elitism is enabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: 27
            reproduction rate: 0.125
            crossover rate: 0.75
            mutation rate: 0.125
        """
        sys.stdout.write("Starting test_elitism: ONE-POINT CROSSOVER, BASIC MUTATION, ELITISM ENABLED\n")

        reproduction = 0.125
        crossover = 0.75
        mutation = 0.125
        population_size = 27

        individual_factory = optimization.XAbsoluteSquareFloatIndividualFactory(crossover_method='one_point', mutation_method='basic_mutation')
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=population_size, reproduction=reproduction, crossover=crossover, mutation=mutation, elitism=True, termination_criteria=termination_criteria)
        
        repeat = 1

        info = pandas.DataFrame([], columns=["generation", "mean", "max", "std", "execution"])
        for idx in xrange(repeat):
            solver.set_params()
            solver.init_population()
            solver.evolve()
            i = solver.get_generation_info()
            i["execution"] = idx
            info = info.append(i)

            matplotlib.pyplot.plot(i['generation'], i['max'], "coral", label="melhor - com elitismo", linewidth=2)
            matplotlib.pyplot.plot(i['generation'], i['mean'], "cadetblue", label="media - com elitismo", linewidth=2)
            matplotlib.pyplot.plot(i['generation'], i['std'], ".", label="desvio - com elitismo")

        info.to_csv('/home/fabio/sin5006/tests/results/xabsolutesquare/test_elitism.csv', sep=',', index=False)

        base = pandas.read_csv('/home/fabio/sin5006/tests/results/xabsolutesquare/test_base.csv')

        matplotlib.pyplot.plot(base["generation"], base["max"], "r", label="melhor - sem elitismo", linewidth=2)
        matplotlib.pyplot.plot(base["generation"], base["mean"], "b", label="media - sem elitismo", linewidth=2)
        matplotlib.pyplot.plot(base["generation"], base["std"], "k.", label="desvio - sem elitismo")

        legend = matplotlib.pyplot.legend(loc='lower right', numpoints=1)
        matplotlib.pyplot.xlabel("geracoes")
        matplotlib.pyplot.ylabel("fitness")
        matplotlib.pyplot.show()

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/xabsolutesquare/test_elitism.csv\n")
        assert True

    def test_uniform_crossover(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: uniform crossover
        Mutation operator: basic (replaces x or y by another valid value)
        Elitism is disabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: 4096
            reproduction rate: 0.125
            crossover rate: 0.75
            mutation rate: 0.125
        """
        sys.stdout.write("Starting test_uniform_crossover: UNIFORM CROSSOVER, BASIC MUTATION, ELITISM DISABLED\n")

        reproduction = 0.125
        crossover = 0.75
        mutation = 0.125
        population_size = 27

        individual_factory = optimization.XAbsoluteSquareFloatIndividualFactory(crossover_method='uniform', mutation_method='basic_mutation')
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=population_size, reproduction=reproduction, crossover=crossover, mutation=mutation, elitism=False, termination_criteria=termination_criteria)

        repeat = 1
        
        info = pandas.DataFrame([], columns=["generation", "mean", "max", "std", "execution"])
        for idx in xrange(repeat):
            solver.set_params()
            solver.init_population()
            solver.evolve()
            i = solver.get_generation_info()
            i["execution"] = idx
            info = info.append(i)

            matplotlib.pyplot.plot(i['generation'], i['max'], "coral", label="melhor - uniform crossover", linewidth=2)
            matplotlib.pyplot.plot(i['generation'], i['mean'], "cadetblue", label="media - uniform crossover", linewidth=2)
            matplotlib.pyplot.plot(i['generation'], i['std'], ".", label="desvio - uniform crossover")
    
        info.to_csv('/home/fabio/sin5006/tests/results/xabsolutesquare/test_uniform_crossover.csv', sep=',', index=False)

        base = pandas.read_csv('/home/fabio/sin5006/tests/results/xabsolutesquare/test_base.csv')

        matplotlib.pyplot.plot(base["generation"], base["max"], "r", label="melhor - one-point crossover", linewidth=2)
        matplotlib.pyplot.plot(base["generation"], base["mean"], "b", label="media - one-point crossover", linewidth=2)
        matplotlib.pyplot.plot(base["generation"], base["std"], "k.", label="desvio - one-point crossover")

        legend = matplotlib.pyplot.legend(loc='lower right', numpoints=1)
        matplotlib.pyplot.xlabel("geracoes")
        matplotlib.pyplot.ylabel("fitness")
        matplotlib.pyplot.show()

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/xabsolutesquare/test_uniform_crossover.csv\n")
        assert True

    def test_permutation_mutation(self):
        """
        Chromosome: float array containing x and y values
        Selection: roulette-wheel
        Crossover operator: uniform crossover
        Mutation operator: permutation
        Elitism is disabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: 27
            reproduction rate: 0.125
            crossover rate: 0.75
            mutation rate: 0.125
        """
        sys.stdout.write("Starting test_permutation_mutation: ONE-POINT CROSSOVER, PERMUTATION MUTATION, ELITISM DISABLED\n")

        reproduction = 0.125
        crossover = 0.75
        mutation = 0.125
        population_size = 27

        individual_factory = optimization.XAbsoluteSquareFloatIndividualFactory(crossover_method='one_point', mutation_method='permutation')
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=population_size, reproduction=reproduction, crossover=crossover, mutation=mutation, elitism=False, termination_criteria=termination_criteria)
        
        repeat = 1
        
        info = pandas.DataFrame([], columns=["generation", "mean", "max", "std", "execution"])
        for idx in xrange(repeat):
            solver.set_params()
            solver.init_population()
            solver.evolve()
            i = solver.get_generation_info()
            i["execution"] = idx
            info = info.append(i)

            matplotlib.pyplot.plot(i['generation'], i['max'], "coral", label="melhor - permutation", linewidth=2)
            matplotlib.pyplot.plot(i['generation'], i['mean'], "cadetblue", label="media - permutation", linewidth=2)
            matplotlib.pyplot.plot(i['generation'], i['std'], ".", label="desvio - permutation")
        
        info.to_csv('/home/fabio/sin5006/tests/results/xabsolutesquare/test_permutation_mutation.csv', sep=',', index=False)

        base = pandas.read_csv('/home/fabio/sin5006/tests/results/xabsolutesquare/test_base.csv')

        matplotlib.pyplot.plot(base["generation"], base["max"], "r", label="melhor - basic mutation", linewidth=2)
        matplotlib.pyplot.plot(base["generation"], base["mean"], "b", label="media - basic mutation", linewidth=2)
        matplotlib.pyplot.plot(base["generation"], base["std"], "k.", label="desvio - basic mutation")

        legend = matplotlib.pyplot.legend(loc='lower right', numpoints=1)
        matplotlib.pyplot.xlabel("geracoes")
        matplotlib.pyplot.ylabel("fitness")
        matplotlib.pyplot.show()

        sys.stdout.write("Finished. Results are at: /home/fabio/sin5006/tests/results/xabsolutesquare/test_permutation_mutation.csv\n")
        assert True

if __name__ == '__main__':
    unittest.main()
