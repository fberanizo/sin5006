# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import unittest, ga, vrp, re, math, numpy

class CVRP(unittest.TestCase):
     """Test cases for CVRP problem."""

     def test_1(self):
        """
        Chromosome: ?
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
        sys.stdout.write("Starting test_1: ONE-POINT CROSSOVER, PERMUTATION MUTATION, ELITISM DISABLED\n")
        sys.stdout.write("Input: ./tests/vrp/A-n32-k5.vrp\n")

        fname = './input/A-n32-k5.vrp'
        content = open(fname).read().replace('\r\n', '')
        nodes = int(re.search('DIMENSION : ([0-9]+)', content).group(1))
        capacity = int(re.search('CAPACITY : ([0-9]+)', content).group(1))
        coords = re.search('NODE_COORD_SECTION\s+(.*)DEMAND_SECTION', content).group(1)
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

        demand_section = re.search('DEMAND_SECTION\s+(.*)DEPOT_SECTION', content).group(1)
        demand_section = re.findall('([0-9]+) ([0-9]+)', demand_section)
        node_demand = dict()
        for node, demand in demand_section:
            node_demand[int(node)] = int(demand)

        reproduction = 0.125
        crossover = 0.75
        mutation = 0.125
        population_size = 27

        individual_factory = vrp.VRPIndividualFactory(nodes, capacity, distances, node_demand, crossover_method='one_point', mutation_method='permutation')
        #individual = individual_factory.create()
        #print individual.genotype
        #print individual.get_fitness()
        termination_criteria = ga.NumberOfGenerationsTerminationCriteria(number_of_generations=100)
        solver = ga.GeneticAlgorithm(individual_factory, population_size=population_size, reproduction=reproduction, crossover=crossover, mutation=mutation, elitism=False, termination_criteria=termination_criteria)
        # solver.set_params()
        solver.init_population()
        solver.evolve()
        print solver.result().get_genotype()
        print solver.result().get_fitness()
        #i = solver.get_generation_info()

        sys.stdout.write("Finished. Results are at: ./tests/vrp/A-n32-k5.vrp\n")
        assert True

if __name__ == '__main__':
    unittest.main()