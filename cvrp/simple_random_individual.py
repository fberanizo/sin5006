# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, cvrp, numpy, struct, math, itertools

class SimpleRandomIndividual(ga.Individual):
    """Represents a solution of a vehicle routing problem."""
    def __init__(self, genotype, fitness_evaluator, distances):
        super(cvrp.SimpleRandomIndividual, self).__init__(genotype, fitness_evaluator)
        self.distances = distances

    def mutate(self):
        """Simple Random Mutation:
           Moves a customer within the same solution.
        """
        genotype = numpy.array(self.genotype, copy=True)
        # Selects and deletes a customer
        p = map(lambda x: 0.0 if x == 'X' else 1.0, genotype)
        s = sum(p)
        k = numpy.random.choice(genotype, p=map(lambda x: x/s, p))
        genotype = filter(lambda x: True if x not in [k] else False, genotype)
        
        # performs best insertion
        c = self.best_insertion(genotype, int(k), int(k))
        genotype = numpy.insert(genotype, c, k)

        return cvrp.SimpleRandomIndividual(genotype, self.fitness_evaluator, self.distances)

    def crossover(self, another_individual):
        """Simple Random Crossover:
           Randomly selects a subroute from P2 and inserts it into P1."""
        child = numpy.array(self.get_genotype(), copy=True)
        another_individual_genotype = another_individual.get_genotype()

        # randomly select a subroute from P2
        routes = [list(y) for x, y in itertools.groupby(another_individual_genotype, lambda z: z == 'X') if not x]
        route = numpy.random.choice(routes)
        indexes = numpy.random.randint(0, len(route), 2)
        subroute = route[min(indexes):max(indexes)+1]

        # delete the members of the subroute from the P1
        child = filter(lambda x: True if x not in subroute else False, child)

        # performs best insertion
        c = self.best_insertion(child, int(subroute[0]), int(subroute[-1]))
        child = numpy.insert(child, c, subroute)

        return cvrp.SimpleRandomIndividual(child, self.fitness_evaluator, self.distances), None

    def best_insertion(self, genotype, k1, kn):
        """Finds out a inserction position for customer k that minimizes replacement cost."""
        costs = []
        for idx in xrange(0, len(genotype)-1):
            m = 1 if genotype[idx] == 'X' else int(genotype[idx])
            m_plus_one = 1 if genotype[idx + 1] == 'X' else int(genotype[idx + 1])
            c = self.distances.item((m-1, m_plus_one-1)) - self.distances.item((m-1, k1-1)) - self.distances.item((kn-1, m_plus_one-1))
            costs.append(c)
        max_val = max(costs)
        max_idx = costs.index(max_val)
        return max_idx


