# -*- coding: utf-8 -*-

import ga, time, numpy

class GeneticAlgorithm(object):
    """Represents a genetic algorigthm structure"""
    def __init__(self, population_size=100):
        self.population_size = population_size
        self.population = []
        self.population_fitness = []
        self.generation_info = []
        self.generation = 0

    def init_population(self, individual_factory):
        """Initialize a population of individuals using a provided function"""
        for idx in xrange(0, 100):
            individual = individual_factory.create()
            self.population.append(individual)

        self.population_fitness = numpy.asarray(map(lambda individual: individual.get_fitness(), self.population))
        s = float(self.population_fitness.sum())
        self.normalized_fitness = numpy.asarray(map(lambda fitness: fitness/s, self.population_fitness))

    def evolve(self, termination_criteria, reproduction=0.15, crossover=0.8, mutation=0.05):
        """Evolves individuals (solutions) until some termination criteria is satisfied"""
        self.generation = 0
        start_time = time.time()

        while not termination_criteria.satisfied(self.generation, time.time()-start_time, self.population):
            self.generation += 1

            # select genetic operation probabilistically
            operations = numpy.random.choice(['reproduction', 'crossover', 'mutation'], size=self.population_size, p=[reproduction, crossover, mutation]).tolist()
            individuals = numpy.random.choice(self.population, p=self.normalized_fitness, size=2*self.population_size, replace=True).tolist()

            next_generation = []
            while len(next_generation) < self.population_size:
                operation = operations.pop()
                individual = individuals.pop()
                individual.get_fitness() # enforce fitness calculation

                if operation == 'reproduction':
                    next_generation.append(individual)
                elif operation == 'crossover':
                    individual2 = individuals.pop()
                    individual2.get_fitness() # enforce fitness calculation
                    individual1, individual2 = individual.crossover(individual2)
                    next_generation.append(individual1)
                    next_generation.append(individual2)
                elif operation == 'mutation':
                    individual.mutate()
                    next_generation.append(individual)

            self.population = next_generation
            self.population_fitness = numpy.asarray(map(lambda individual: individual.get_fitness(), self.population))
            s = float(self.population_fitness.sum())
            self.normalized_fitness = numpy.asarray(map(lambda fitness: fitness/s, self.population_fitness))

            self.generation_info.append({
                "avg": -numpy.mean(self.population_fitness),
                "std": -numpy.std(self.population_fitness),
                "max": -self.population_fitness.max()
            })

            #print "Generation: " + str(self.generation)
            #print "Best solution: " + str(self.population[0].get_genotype()) + " => " + str(self.population[0].get_fitness())

    def result(self):
        """Returns one best solution"""
        return max(self.population, key=lambda individual: individual.get_fitness())

    def get_generation_info(self):
        return self.generation_info