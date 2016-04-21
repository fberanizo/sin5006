# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import unittest, ga_test_case, ga, optimization, time, matplotlib.pyplot

class Case1(ga_test_case.GATestCase):
    """
    Chromosome: array containing x and y values
    Selection: roulette-wheel
    Crossover operator: one-point crossover
    Mutation operator: basic (replaces x or y by another valid value)
    Termination criteria: number of generations = 100

    Parameters:
        population_size: 300
        crossover rate: 0.7
        reproduction rate: 0.2
        mutation rate: 0.1
    """

    def test_case1(self):
        start_time = time.time()

        solver = ga.GeneticAlgorithm(population_size=300)
        solver.init_population(optimization.IndividualFactory(crossover_method='one-point'))
        solver.evolve(ga.NumberOfGenerationsTerminationCriteria(), reproduction=0.2, crossover=0.7, mutation=0.1)
        

        individual = solver.result()
        genotype = individual.get_genotype()
        generation_info = solver.get_generation_info()

        #super.plot(generation_info)

        print 'Execution time: ' + str(time.time() - start_time)
        print 'Number of generations: ' + str(solver.generation)
        print 'x = ' + str(genotype[0]) + ', y = ' + str(genotype[1])
        print 'fitness = ' + str(individual.get_fitness()) 
        assert True


if __name__ == '__main__':
    unittest.main()

#uso de diferentes tamanhos de população, de grandezes bem diferentes;
#mais de um critério de parada;
#para a parte I - cromossomo com codificação binária e real, a depender da função sob otimização;
#um operador de seleção (preferencialmente roleta);
#dois operadores de crossover (crossover de um ponto e outro à escolha do aluno);
#dois operadores de mutação (mutação simples e outro à escolha do aluno);
#dois critérios de troca de população;
#evolução sem eletismo e com elitismo.
