import os
import random

import orekit
import numpy as np
from deap import creator, base
from deap.benchmarks import tools

from src.Coverage.walker import Walker
from src.Coverage.CoverageDefinition import CoverageDefinition
from src.Coverage.FieldOfViewEventAnalysis import FieldOfViewEventAnalysis
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from orekit.pyhelpers import setup_orekit_curdir
from src.Design import Design
from src.Evaluator import Evaluator
import os
from src.Design import Design
from src.Evaluator import Evaluator
import random
from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as time


os.chdir("C:/Users/lmbed/PycharmProjects/401tradespace/")

vm = orekit.initVM()
setup_orekit_curdir()
epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, TimeScalesFactory.getUTC()) # AbsoluteDate(int year, int month, int day, int hour, int minute, double second, TimeScale timeScale)
# type = "delta"
# altitude = 900 #km
# i = 60 #degrees
# semiMajorAxis = (altitude+6378) * 1000
# w = Walker("One", semiMajorAxis, np.radians(i), 32, 4, 1, epochDate, type)
# w.createConstellation()
# cdef = CoverageDefinition("One", np.radians(10), w)
# fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
# coverage = fovanal.call()
# print(coverage)
type = "delta"
altitude = 1000 #km
i = 90 #degrees
semiMajorAxis = (altitude+6378) * 1000
f = 1

w = Walker("Star", semiMajorAxis, np.radians(i), 32, 5, 2, epochDate, type)
w.createConstellation()
cdef = CoverageDefinition("One", np.radians(10), w)
print(cdef.numPoints)
fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
coverage = fovanal.call()
print(coverage)
print(sum(coverage) / len(coverage))
print()
w = Walker("Star", semiMajorAxis, np.radians(i), 32, 5, 2.25, epochDate, type)
w.createConstellation()
cdef = CoverageDefinition("One", np.radians(10), w)
print(cdef.numPoints)
fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
coverage = fovanal.call()
print(coverage)
print(sum(coverage) / len(coverage))
print()
w = Walker("Star", semiMajorAxis, np.radians(i), 32, 5, 2.5, epochDate, type)
w.createConstellation()
cdef = CoverageDefinition("One", np.radians(10), w)
print(cdef.numPoints)
fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
coverage = fovanal.call()
print(coverage)
print(sum(coverage) / len(coverage))
print()
w = Walker("Star", semiMajorAxis, np.radians(i), 32, 5, 2.75, epochDate, type)
w.createConstellation()
cdef = CoverageDefinition("One", np.radians(10), w)
print(cdef.numPoints)
fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
coverage = fovanal.call()
print(coverage)
print(sum(coverage) / len(coverage))
print()
w = Walker("Star", semiMajorAxis, np.radians(i), 32, 5, 3 , epochDate, type)
w.createConstellation()
cdef = CoverageDefinition("One", np.radians(10), w)
print(cdef.numPoints)
fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
coverage = fovanal.call()
print(coverage)
print(sum(coverage) / len(coverage))
print()
# w = Walker("Star", semiMajorAxis, np.radians(i), 24, 4, f, epochDate, type)
# w.createConstellation()
# w2 = Walker("Delta", semiMajorAxis, np.radians(0.1), 8, 1, f, epochDate, type)
# w2.createConstellation()
# w.combineWalkers(w2)
#
# cdef = CoverageDefinition("One", np.radians(10), w)
# print(cdef.numPoints)
# fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
# coverage = fovanal.call()
# print(coverage)
# print(sum(coverage) / len(coverage))
# print()

# w = Walker("One", semiMajorAxis, np.radians(i), 32, 4, 1, epochDate, type)
# w.createConstellation()
# cdef = CoverageDefinition("One", np.radians(10), w)
# fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
# coverage = fovanal.call()
# print(coverage)
# w = Walker("One", semiMajorAxis, np.radians(i), 32, 4, 2, epochDate, type)
# w.createConstellation()
# cdef = CoverageDefinition("One", np.radians(10), w)
# fovanal = FieldOfViewEventAnalysis([cdef], 90.0)
# coverage = fovanal.call()
# print(coverage)


# Parameters
granularity = 10.0 # Spacing of groundpoints on globe
fov_half_angle = 90.0 # half angle of recieving ground antenna on satellite
epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, TimeScalesFactory.getUTC()) # AbsoluteDate(int year, int month, int day, int hour, int minute, double second, TimeScale timeScale)
parameters = [fov_half_angle, granularity, epochDate]

# Test variables
# variables = [(altitude+6378)*1000.0, np.radians(i), 20, 3, 1]
#
# design = Design("test1", parameters, variables)
# cov = design.getCoverage()
# print(cov)
# variables2 = [(altitude+6378)*1000.0, np.radians(i), 2, 1, 1]
# design2 = Design("test2", parameters, variables2)
# cov2 = design2.getCoverage()
# print(cov2)
# Variable randomizers for optimization
# a, i, n, p, f
def a():
    """
    :return: semimajor axis in meters
    """
    h = random.randrange(500, 1000, 10)
    return float((h + 6378) * 1000)

def i():
    """
       :return: inclination in radians
    """
    return float(np.radians(float(random.randrange(0, 900, 1))/10.0))

def n():
    """
    :return: number of satellites in constellation
    """
    return int(random.randrange(10, 32))

def p():
    """
    :return: number of planes in walker-delta constellation
    """
    return int(random.randrange(2, 8, 1))

def f():
    """
    :return: relative angular spacing between satellites in adjacent planes
    """
    return float(random.randrange(0, 360, 1))


def evaluate(individual):
    variables = individual
    print(variables)
    design = Design("name", parameters, variables)
    coverage_percent = design.getCoveragePercent()
    planes = design.numplanes
    planes_norm = float(planes)/8.0
    return coverage_percent, planes

def customMutation(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            if i == 0:
                individual[i] = a()
            elif i == 1:
                individual[i] = i()
            elif i == 2:
                individual[i] = n()
            elif i == 3:
                individual[i] = p()
            elif i == 4:
                individual[i] = f()

    return individual,

##############################################################################
############## # # # # Implement Evolutionary Algorithm # # # # ##############


creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # maximize the design vector, minimize the cost
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

BOUND_LOW, BOUND_UP = 0.0, 1.0
# Attribute Generator
# toolbox.register("design", design)
toolbox.register("a", a)
toolbox.register("i", i)
toolbox.register("n", n)
toolbox.register("p", p)
toolbox.register("f", f)



toolbox.register("individual", tools.initCycle, creator.Individual,
                 [toolbox.a, toolbox.i, toolbox.n,
                  toolbox.p, toolbox.f], n=1)
toolbox.register("customMut", customMutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.2)
toolbox.register("mutate", toolbox.customMut, indpb=.2)
toolbox.register("select", tools.selNSGA2)


def nsgaii(seed=None):
    random.seed(seed)

    NGEN = 1
    MU = 100
    CXPB = 0.8

    # Normalization Reference Values
    cost_norm_low, cost_norm_high = 130000, 30000
    life_norm_low, life_norm_high = 1020300, 860000
    score_norm_low, score_norm_high = 1, 1.6

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    pop = toolbox.population(n=MU)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # Normalize Objectives
    # for ind in pop:
    #     nnscore, nncost, nnlifetime = ind.fitness.values
    #     nscore = 1 * (nnscore - score_norm_low) / (score_norm_high - score_norm_low)
    #     ncost = 1 * (nncost - cost_norm_low) / (cost_norm_high - cost_norm_low)
    #     nlifetime = 1 * (nnlifetime - life_norm_low) / (life_norm_high - life_norm_low)
    #     ind.fitness.values = nscore, ncost, nlifetime
    generations = [pop]  # save each generation
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Normalize Objectives
        # for ind in pop:
        #     nnscore, nncost, nnlifetime = ind.fitness.values
        #     nscore = 1 * (nnscore - score_norm_low) / (score_norm_high - score_norm_low)
        #     ncost = 1 * (nncost - cost_norm_low) / (cost_norm_high - cost_norm_low)
        #     nlifetime = 1 * (nnlifetime - life_norm_low) / (life_norm_high - life_norm_low)
        #     ind.fitness.values = nscore, ncost, nlifetime
        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        generations.append(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    # print("Final population hypervolume is %f" % hypervolume(pop))

    return pop, logbook, generations

# pop, logbook, generations = nsgaii()