# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, vrp, numpy, struct, math

class VRPIndividualFactory(ga.IndividualFactory):
    def __init__(self, nodes, capacity, distances, demand, crossover_method='one_point', mutation_method='permutation'):
        super(vrp.VRPIndividualFactory, self).__init__()
        self.nodes = nodes
        self.capacity = capacity
        self.distances = distances
        self.demand = demand
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

    def create(self):
        """Creates individuals of this type: [customer1,..,X,customer3,customer4,...,X,...]."""
        genotype = []

        demanding_nodes = numpy.random.choice(numpy.arange(1, self.nodes+1), replace=False, size=self.nodes)
        vehicle = 0
        for node in demanding_nodes:
            demand = self.demand[node]
            # se a próxima demanda excede a capacidade, aloca outro veículo
            if vehicle + demand > self.capacity:
                genotype.append('X')
                vehicle = 0
            vehicle += demand
            genotype.append(node)
        genotype = numpy.asarray(genotype)

        fitness_evaluator = vrp.VRPFitnessEvaluator(self.nodes, self.capacity, self.distances, self.demand)
        return vrp.Individual(genotype, fitness_evaluator, self.crossover_method, self.mutation_method)

ga.IndividualFactory.register(VRPIndividualFactory)