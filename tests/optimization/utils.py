# -*- coding: utf-8 -*-

import numpy, matplotlib.pyplot, pandas, seaborn

def plot(execution_info, title='', description=''):

    for generation_info in execution_info:
        x = numpy.arange(1, len(generation_info)+1)
        max = numpy.asarray(map(lambda individual: individual["max"], generation_info))
        avg = numpy.asarray(map(lambda individual: individual["avg"], generation_info))
        std = numpy.asarray(map(lambda individual: individual["std"], generation_info))

        matplotlib.pyplot.plot(x, max, "r", label="melhor", linewidth=1)
        matplotlib.pyplot.plot(x, avg, "b", label="media", linewidth=1)
        matplotlib.pyplot.plot(x, std, "k.", label="desvio")

    matplotlib.pyplot.xlabel('generations')
    matplotlib.pyplot.ylabel('fitness')

    #legend = matplotlib.pyplot.legend(loc='lower right')
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.figtext(.02, .02, description)
    matplotlib.pyplot.gca().set_position((.1, .3, .8, .6))

    matplotlib.pyplot.show()

def save_scores(filepath, grid_scores):
    f = open(filepath, "w")
    f.write(",".join(["Population", "Operators", "Fitness", "FitnessStdDev"]) + "\n")
    for score in grid_scores:
        mean_best_fitness = "{:.6f}".format(score["mean_best_fitness"])
        std_best_fitness = "{:.6f}".format(score["std_best_fitness"])
        population_size = str(score["params"]["population_size"])
        reproduction = "{:.3f}".format(score["params"]["operators_rate"][0])
        crossover = "{:.3f}".format(score["params"]["operators_rate"][1])
        mutation = "{:.3f}".format(score["params"]["operators_rate"][2])
        fields = [population_size, "\"R: "+reproduction+", C: "+crossover +", M: "+mutation+"\"", mean_best_fitness, std_best_fitness]
        f.write(",".join(fields) + "\n")
    f.close()

def plot_heatmap(filepath, dataset):
    seaborn.set()
    h = seaborn.heatmap(dataset, annot=True, linewidths=.5)
    seaborn.plt.yticks(rotation=0)
    seaborn.plt.show()
    #seaborn.plt.savefig(filepath)