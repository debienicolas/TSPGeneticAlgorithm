import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import Reporter
import numpy as np
import random

meanObjectives = []
bestObjectives = []
bestSolutions = []
diversityScores = []
iterations = []


# Modify the class name to match your student number.
class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.iter = 0
        self.population = None



        


    


    # The evolutionary algorithm's main loop
    def optimize(self, filename,p):
        # Read distance matrix from file.
        distanceMatrix = None
        with open(filename) as file:
            distanceMatrix = np.loadtxt(file, delimiter=",")
        # Your code here.

        # Create a TSP problem instance.
        tsp = TSPProblem(distanceMatrix)

        ### Population initialization
       
        self.iter = p.num_iters
        self.initializePopulation(tsp, p)


        while(self.convergenceTest()):
            
            ### Create offspring population
            offspring = np.zeros(p.mu, dtype=Individual)
            for i in range(p.mu):
                p1 = self.selection(tsp,p)
                p2 = self.selection(tsp,p)
                offspring[i] = self.recombination(tsp, p1,p2)
                self.mutation(offspring[i])
            
            ### Mutate the original population
            for ind in self.population:
                self.mutation(ind)
            
            ### Elimination
            self.population = self.elimination(tsp, offspring,p)

            
            fitnesses = [self.fitness(tsp,c) for c in self.population]
            meanObjective = np.mean(fitnesses)
            bestObjective = np.min(fitnesses)
            bestSolution = self.population[np.argmin(fitnesses)].order
            diversityScore = len(set(fitnesses))

            print("Mean objective: ", round(meanObjective,2),"     Best Objective: ", round(bestObjective,2),"     Iterations to go: ",self.iter, "     Diversity score: ", diversityScore)
            meanObjectives.append(meanObjective)
            bestObjectives.append(bestObjective)
            diversityScores.append(diversityScore)

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        # Your code here.
        return bestObjective


class Individual:
    def __init__(self, tsp, order=None, alpha=None):
        self.alpha = max(0.1, 0.1 + 0.2 * random.random())
        self.order = np.random.permutation(range(tsp.nCities))
        self.size = len(self.order)

        # If you don't want individual to be random
        if order is not None:
            self.order = order
            self.size = len(self.order)
        if alpha is not None:
            self.alpha = max(alpha,alpha + 0.25 * random.random())



class TSPProblem:
    def __init__(self, distanceMatrix):
        self.distanceMatrix = distanceMatrix
        self.nCities = len(distanceMatrix)


class Parameters:
    def __init__(self, lamb, mu, num_iters, k,alpha):
        self.lamb = lamb  # Population size
        self.mu = mu  # Amount of offspring
        self.num_iters = num_iters  # How many times to create offspring
        self.k = k  # k for k-tournament selection
        self.alpha = alpha # chance of mutation


if __name__ == "__main__":
    # p = Parameters(lamb=100, mu=500,num_iters=500,k=12,alpha=0.2)
    # algo = r0123456()
    # best = algo.optimize("./tour50.csv",p)

    runs = 15
    bestObject = np.zeros(runs)
    print(bestObject.shape)
    for i in range(runs):
        p = Parameters(lamb=100, mu=500,num_iters=500,k=12,alpha=0.15)
        algo = r0123456()
        best = algo.optimize("./tour50.csv",p)
        bestObject[i] = best
    
    print(bestObject.mean())
    print(bestObject.std())

    print(bestObject)

