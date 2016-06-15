# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, vrp, numpy, struct, math, sklearn.preprocessing

class SimpleRandomIndividual(ga.Individual):
    """Represents a solution of a vehicle routing problem."""
    def __init__(self, genotype, fitness_evaluator, distances):
        super(vrp.Individual, self).__init__(genotype, fitness_evaluator)
        self.distances = distances

    def mutate(self):
        """Simple Random Mutation:
           Moves a customer within the same solution.
        """
        genotype = numpy.array(individual.genotype, copy=True)
        # Selects and deletes a customer
        p = map(lambda x: 0.0 if x == 'X' else 1.0, genotype)
        s = sum(p)
        idx = numpy.random.choice(genotype, p=map(lambda x: x/s, p))
        k = genotype[idx]
        numpy.delete(genotype, idx)
        
        # finds out best insertion index
        c = self.best_insertion(genotype, k, k)
        numpy.insert(genotype, c, k)

        return vrp.SimpleRandomIndividual(genotype, individual.fitness_evaluator, self.distances)

    def best_insertion(self, genotype, k1, kn):
        """Finds out a inserction position for customer k that minimizes replacement cost."""
        costs = []
        for idx in xrange(0, len(genotype)-1):
            m = 0 if genotype[idx] == 'X' else genotype[idx]
            m_plus_one = 0 if genotype[idx + 1] == 'X' else genotype[idx + 1]
            c = self.distances.item((m, m_plus_one)) - self.distances.item((m, k1)) - self.distances.item((kn, m_plus_one))
            costs[idx] = c
        max_val = max(costs)
        max_idx = costs.index(max_val)
        return max_idx

    def crossover(self, another_individual):
        """Simple Random Crossover:
           Randomly selects a subroute from P2 and inserts it into P1."""
        child = numpy.array(self.genotype, copy=True)

        routes = [list(y) for x, y in itertools.groupby(another_individual, lambda z: z == 'X') if not x]
        route = numpy.random.choice(routes)
        indexes = numpy.random.randint(0, len(route), 2)
        subroute = route[min(indexes):max(indexes)+1]

        numpy.delete(child, subroute)

        c = best_insertion(child, subroute[0], subroute[-1])
        numpy.insert(child, c, subroute)

        return vrp.SimpleRandomIndividual(child, individual.fitness_evaluator, self.distances), None


