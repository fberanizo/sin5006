# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import unittest

class GATestCase(unittest.TestCase):
    def plot(self, generation_info):
        x = numpy.arange(1, len(info)+1)
        avg = numpy.asarray(map(lambda individual: individual["avg"], info))
        std = numpy.asarray(map(lambda individual: individual["std"], info))
        max = numpy.asarray(map(lambda individual: individual["max"], info))

        matplotlib.pyplot.xlabel('generations')
        matplotlib.pyplot.ylabel('fitness')

        matplotlib.pyplot.plot(x, avg, "g", label="mean", linewidth=1.2)
        matplotlib.pyplot.plot(x, std, "r", label="std", linewidth=1.2)
        matplotlib.pyplot.plot(x, max, "b", label="max", linewidth=1.2)
        #pylab.plot(x, std_dev_y, "k", label="Std Dev raw", linewidth=1.2)

        matplotlib.pyplot.show()
