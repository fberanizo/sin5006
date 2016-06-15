# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import unittest, ga, cvrp, re, math, numpy, matplotlib.pyplot as plt

class ClassicalOperators(unittest.TestCase):
    """Test cases for CVRP problem."""

    def test_1(self):
        """
        Chromosome: sequence of clients separed by an 'X' when a new vehicle is assigned
        Selection: roulette-wheel
        Crossover operator: one-point
        Mutation operator: permutation
        Elitism is enabled
        Termination criteria: number of generations = 1000
        Parameters:
            population_size: 97
            reproduction rate: 0.5
            crossover rate: 0
            mutation rate: 0.5
        """
        sys.stdout.write("Starting test_1: CLASSICAL OPERATORS, ELITISM ENABLED\n")
        sys.stdout.write("Input: ./tests/cvrp/A-n32-k5.vrp\n")

        fname = './input/A-n32-k5.vrp'
        nodes, capacity, distances, demand = self.load_test(fname)

        reproduction = 0.5
        crossover = 0
        mutation = 0.5
        population_size = 97

        individual_factory = cvrp.CVRPIndividualFactory(nodes, capacity, distances, demand, 'classical')
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=1000)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=population_size, reproduction=reproduction, crossover=crossover, mutation=mutation, elitism=True, termination_criteria=termination_criteria)
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
        Termination criteria: number of generations = 1000
        Parameters:
            population_size: 97
            reproduction rate: 0.5
            crossover rate: 0
            mutation rate: 0.5
        """
        sys.stdout.write("Starting test_2: CLASSICAL OPERATORS, ELITISM ENABLED\n")
        sys.stdout.write("Input: ./tests/vrp/B-n31-k5.vrp\n")

        fname = './input/B-n31-k5.vrp'
        nodes, capacity, distances, demand = self.load_test(fname)

        reproduction = 0.5
        crossover = 0
        mutation = 0.5
        population_size = 97

        individual_factory = cvrp.CVRPIndividualFactory(nodes, capacity, distances, demand)
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=1000)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=population_size, reproduction=reproduction, crossover=crossover, mutation=mutation, elitism=True, termination_criteria=termination_criteria)
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
        Termination criteria: number of generations = 1000
        Parameters:
            population_size: 97
            reproduction rate: 0.5
            crossover rate: 0
            mutation rate: 0.5
        """
        sys.stdout.write("Starting test_3: CLASSICAL OPERATORS, ELITISM ENABLED\n")
        sys.stdout.write("Input: ./tests/vrp/P-n16-k8.vrp\n")

        fname = './input/P-n16-k8.vrp'
        nodes, capacity, distances, demand = self.load_test(fname)

        reproduction = 0.5
        crossover = 0
        mutation = 0.5
        population_size = 97

        individual_factory = cvrp.CVRPIndividualFactory(nodes, capacity, distances, demand)
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=1000)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=population_size, reproduction=reproduction, crossover=crossover, mutation=mutation, elitism=True, termination_criteria=termination_criteria)
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
        plt.show()

        sys.stdout.write("Finished. Results are at: ./results/classical_operators/P-n16-k8.vrp.csv\n")
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