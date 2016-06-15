# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, cvrp, numpy, struct, math

class CVRPIndividualFactory(ga.IndividualFactory):
    def __init__(self, nodes, capacity, distances, demand):
        super(cvrp.CVRPIndividualFactory, self).__init__()
        self.nodes = nodes
        self.capacity = capacity
        self.distances = distances
        self.demand = demand

    def create(self, individual_type='classical'):
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

        fitness_evaluator = cvrp.CVRPFitnessEvaluator(self.nodes, self.capacity, self.distances, self.demand)
        
        if individual_type == 'classical':
            return cvrp.ClassicalIndividual(genotype, fitness_evaluator)
        elif individual_type == 'simple_random':
            return cvrp.SimpleRandomIndividual(genotype, fitness_evaluator, distances)

ga.IndividualFactory.register(CVRPIndividualFactory)