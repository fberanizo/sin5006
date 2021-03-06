# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, cvrp, numpy

class CVRPFitnessEvaluator(ga.FitnessEvaluator):
    def __init__(self, nodes, capacity, distances, demand):
        super(cvrp.CVRPFitnessEvaluator, self).__init__()
        self.nodes = nodes
        self.capacity = capacity
        self.distances = distances
        self.demand = demand

    def evaluate(self, individual):
        """Evaluates individual based on minimal route distance summation."""
        fitness = 0
        capacity = self.capacity
        previous = 1 # vehicle starts at depot
        i = 0
        feasible = True
        #print individual.get_genotype()
        while i < len(individual.get_genotype()):
            current = individual.get_genotype()[i]
            if current == 'X': # go back to depot
                fitness += self.distances.item((previous-1, 0))
                previous = 1 # next vehicle also starts at depot
                capacity = self.capacity # with full capaity
            else: # visit this node
                current = int(current)
                #print type(previous)
                #print current
                #print self.distances.item((previous-1, current-1))
                fitness += self.distances.item((previous-1, current-1))
                capacity -= self.demand[current]
                # if demand is greater than vehicle capacity, fitness is made too big
                if capacity < 0:
                    feasible = False
                previous = current
            i += 1
        fitness += self.distances.item((previous-1, 0))

        if not feasible:
            fitness = self.distances.sum()

        return fitness


ga.FitnessEvaluator.register(CVRPFitnessEvaluator)