# -*- coding: utf-8 -*-

import numpy, matplotlib.pyplot

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
