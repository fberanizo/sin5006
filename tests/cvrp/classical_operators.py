# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import unittest, ga, cvrp, grid_search, re, math, numpy, itertools, matplotlib.pyplot as plt

class ClassicalOperators(unittest.TestCase):
    """Test cases for CVRP problem."""
    grid_search = False

    def test_1(self):
        """
        Chromosome: sequence of clients separed by an 'X' when a new vehicle is assigned
        Selection: roulette-wheel
        Crossover operator: one-point
        Mutation operator: permutation
        Elitism is enabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: 4096
            reproduction rate: 0.375
            crossover rate: 0
            mutation rate: 0.625
        """
        fname = './input/A-n32-k5.vrp'
        nodes, capacity, distances, demand = self.load_test(fname)

        individual_factory = cvrp.CVRPIndividualFactory(nodes, capacity, distances, demand, 'classical')
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=4096, reproduction=0.375, crossover=0.0, mutation=0.625, elitism=True, termination_criteria=termination_criteria)

        if self.grid_search:
            params = {
                "population_size": numpy.logspace(3, 12, base=2, num=6, dtype=int),
                "operators_rate": map(lambda x: (x[0], 0.0, x[1]), filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.125, 0.875, .125), repeat=2))),
                "elitism": [True],
                "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
            }
            grid = grid_search.GridSearch(solver, params)
            grid.search(0.0)
            grid_scores = grid.get_grid_scores()
            
            fname = './results/classical_operators/A-n32-k5.vrp.grid.csv'
            grid_scores.to_csv(fname, sep=',', index=False)
            sys.stdout.write("Finished. Results are at: ./results/classical_operators/A-n32-k5.vrp.grid.csv\n")
        else:
            sys.stdout.write("Starting test_1: CLASSICAL OPERATORS, ELITISM ENABLED\n")
            sys.stdout.write("Input: ./tests/cvrp/A-n32-k5.vrp\n")
            
            solver.init_population()
            solver.evolve()

            info = solver.get_generation_info()
            fname = './results/classical_operators/A-n32-k5.vrp.csv'
            info.to_csv(fname, sep=',', index=False)

            plt.plot(info['generation'], info['min'], "r", label="melhor", linewidth=2)
            plt.plot(info['generation'], info['mean'], "b", label="media", linewidth=2)
            plt.plot(info['generation'], info['std'], "k.", label="desvio")

            legend = plt.legend(loc='lower right', numpoints=1)
            plt.xlabel("geracoes")
            plt.ylabel("fitness")
            plt.yscale('log')
            plt.show()

            sys.stdout.write("Finished. Results are at: ./results/classical_operators/A-n32-k5.vrp.csv\n")
        assert True

    def test_2(self):
        """
        Chromosome: sequence of clients separed by an 'X' when a new vehicle is assigned
        Selection: roulette-wheel
        Crossover operator: one-point
        Mutation operator: permutation
        Elitism is enabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: 4096
            reproduction rate: 0.375
            crossover rate: 0
            mutation rate: 0.625
        """
        fname = './input/B-n31-k5.vrp'
        nodes, capacity, distances, demand = self.load_test(fname)

        individual_factory = cvrp.CVRPIndividualFactory(nodes, capacity, distances, demand)
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=4096, reproduction=0.375, crossover=0.0, mutation=0.625, elitism=True, termination_criteria=termination_criteria)

        if self.grid_search:
            params = {
                "population_size": numpy.logspace(3, 12, base=2, num=6, dtype=int),
                "operators_rate": map(lambda x: (x[0], 0.0, x[1]), filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.125, 0.875, .125), repeat=2))),
                "elitism": [True],
                "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
            }
            grid = grid_search.GridSearch(solver, params)
            grid.search(0.0)
            grid_scores = grid.get_grid_scores()
            
            fname = './results/classical_operators/B-n31-k5.vrp.grid.csv'
            grid_scores.to_csv(fname, sep=',', index=False)
            sys.stdout.write("Finished. Results are at: ./results/classical_operators/B-n31-k5.vrp.grid.csv\n")
        else:
            sys.stdout.write("Starting test_2: CLASSICAL OPERATORS, ELITISM ENABLED\n")
            sys.stdout.write("Input: ./tests/vrp/B-n31-k5.vrp\n")

            solver.init_population()
            solver.evolve()
            info = solver.get_generation_info()
            fname = './results/classical_operators/B-n31-k5.vrp.csv'
            info.to_csv(fname, sep=',', index=False)

            plt.plot(info['generation'], info['min'], "r", label="melhor", linewidth=2)
            plt.plot(info['generation'], info['mean'], "b", label="media", linewidth=2)
            plt.plot(info['generation'], info['std'], "k.", label="desvio")

            legend = plt.legend(loc='lower right', numpoints=1)
            plt.xlabel("geracoes")
            plt.ylabel("fitness")
            plt.yscale('log')
            plt.show()

            sys.stdout.write("Finished. Results are at: ./results/classical_operators/B-n31-k5.vrp.csv\n")
        assert True

    def test_3(self):
        """
        Chromosome: sequence of clients separed by an 'X' when a new vehicle is assigned
        Selection: roulette-wheel
        Crossover operator: one-point
        Mutation operator: permutation
        Elitism is enabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: 4096
            reproduction rate: 0.375
            crossover rate: 0
            mutation rate: 0.625
        """
        fname = './input/P-n16-k8.vrp'
        nodes, capacity, distances, demand = self.load_test(fname)

        individual_factory = cvrp.CVRPIndividualFactory(nodes, capacity, distances, demand)
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=4096, reproduction=0.375, crossover=0.0, mutation=0.625, elitism=True, termination_criteria=termination_criteria)

        if self.grid_search:
            params = {
                "population_size": numpy.logspace(3, 12, base=2, num=6, dtype=int),
                "operators_rate": map(lambda x: (x[0], 0.0, x[1]), filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.125, 0.875, .125), repeat=2))),
                "elitism": [True],
                "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
            }
            grid = grid_search.GridSearch(solver, params)
            grid.search(0.0)
            grid_scores = grid.get_grid_scores()
            
            fname = './results/classical_operators/P-n16-k8.vrp.grid.csv'
            grid_scores.to_csv(fname, sep=',', index=False)
            sys.stdout.write("Finished. Results are at: ./results/classical_operators/P-n16-k8.vrp.grid.csv\n")
        else:
            sys.stdout.write("Starting test_3: CLASSICAL OPERATORS, ELITISM ENABLED\n")
            sys.stdout.write("Input: ./tests/vrp/P-n16-k8.vrp\n")

            solver.init_population()
            solver.evolve()
            info = solver.get_generation_info()
            fname = './results/classical_operators/P-n16-k8.vrp.csv'
            info.to_csv(fname, sep=',', index=False)

            plt.plot(info['generation'], info['min'], "r", label="melhor", linewidth=2)
            plt.plot(info['generation'], info['mean'], "b", label="media", linewidth=2)
            plt.plot(info['generation'], info['std'], "k.", label="desvio")

            legend = plt.legend(loc='lower right', numpoints=1)
            plt.xlabel("geracoes")
            plt.ylabel("fitness")
            plt.yscale('log')
            plt.show()

            sys.stdout.write("Finished. Results are at: ./results/classical_operators/P-n16-k8.vrp.csv\n")
        assert True

    def test_4(self):
        """
        Chromosome: sequence of clients separed by an 'X' when a new vehicle is assigned
        Selection: roulette-wheel
        Crossover operator: one-point
        Mutation operator: permutation
        Elitism is enabled
        Termination criteria: number of generations = 100
        Parameters:
            population_size: 4096
            reproduction rate: 0.25
            crossover rate: 0
            mutation rate: 0.75
        """
        fname = './input/A-n80-k10.vrp'
        nodes, capacity, distances, demand = self.load_test(fname)

        individual_factory = cvrp.CVRPIndividualFactory(nodes, capacity, distances, demand)
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=4096, reproduction=0.25, crossover=0.0, mutation=0.75, elitism=True, termination_criteria=termination_criteria)

        if self.grid_search:
            params = {
                "population_size": numpy.logspace(3, 12, base=2, num=6, dtype=int),
                "operators_rate": map(lambda x: (x[0], 0.0, x[1]), filter(lambda x: sum(x) == 1.0, itertools.product(numpy.arange(.125, 0.875, .125), repeat=2))),
                "elitism": [True],
                "termination_criteria": [ ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100) ]
            }
            grid = grid_search.GridSearch(solver, params)
            grid.search(0.0)
            grid_scores = grid.get_grid_scores()
            
            fname = './results/classical_operators/A-n80-k10.vrp.grid.csv'
            grid_scores.to_csv(fname, sep=',', index=False)
            sys.stdout.write("Finished. Results are at: ./results/classical_operators/A-n80-k10.vrp.grid.csv\n")
        else:
            sys.stdout.write("Starting test_4: CLASSICAL OPERATORS, ELITISM ENABLED\n")
            sys.stdout.write("Input: ./tests/vrp/A-n80-k10.vrp\n")

            solver.init_population()
            solver.evolve()
            info = solver.get_generation_info()
            fname = './results/classical_operators/A-n80-k10.vrp.csv'
            info.to_csv(fname, sep=',', index=False)

            plt.plot(info['generation'], info['min'], "r", label="melhor", linewidth=2)
            plt.plot(info['generation'], info['mean'], "b", label="media", linewidth=2)
            plt.plot(info['generation'], info['std'], "k.", label="desvio")

            legend = plt.legend(loc='lower right', numpoints=1)
            plt.xlabel("geracoes")
            plt.ylabel("fitness")
            plt.yscale('log')
            plt.show()

            sys.stdout.write("Finished. Results are at: ./results/classical_operators/A-n80-k10.vrp.csv\n")
        assert True


    def load_test(self, fname):
        content = open(fname).read().replace('\r', ' ').replace('\n', ' ')
        nodes = int(re.search('DIMENSION : ([0-9]+)', content).group(1))
        capacity = int(re.search('CAPACITY : ([0-9]+)', content).group(1))
        coords = re.search('NODE_COORD_SECTION\s*(.*)\s*DEMAND_SECTION', content).group(1)
        coords = re.findall('([0-9]+) ([0-9]+) ([0-9]+)', coords)
        distances = []
        for node, x1, y1 in coords:
            row = []
            for node, x2, y2 in coords:
                xd = int(x2) - int(x1)
                yd = int(y2) - int(y1)
                row.append(math.sqrt(xd*xd + yd*yd))
            distances.append(row)
        distances = numpy.matrix(distances)

        demand_section = re.search('DEMAND_SECTION\s*(.*)\s*DEPOT_SECTION', content).group(1)
        demand_section = re.findall('([0-9]+) ([0-9]+)', demand_section)
        node_demand = dict()
        for node, demand in demand_section:
            node_demand[int(node)] = int(demand)

        return nodes, capacity, distances, node_demand

    
if __name__ == '__main__':
    unittest.main()