from hmac import new
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import Reporter
import numpy as np
import random
import time
import multiprocessing
from timeit import default_timer as timer
from numba import jit, njit
import cProfile
import pstats

# Modify the class name to match your student number.
class r0123456:
    def __init__(self,p):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.iter = 0
        self.population = None
        self.p = p
        self.timings = {}
    
    @njit
    def initializePopulation(self):
        # create the initial population with random np.permutations
        population = np.zeros((self.p.lamb,self.length),dtype=int)
        population = (lambda x: np.random.permutation(range(self.length)))(population)
        return population

    @njit
    def convergenceTest(self):
        self.iter -= 1
        return self.iter > 0
    
    @njit
    def fitness(self,individual):
        # calculate the fitness of the individual
        # the fitness is the total distance of the cycle
        indices = np.stack((individual,np.roll(individual,-1)),axis=1)
        return np.sum(self.distanceMatrix[indices[:,0],indices[:,1]])
    
    @njit
    def selection(self,population):
        # k-tournament selection
        chosen = population[np.random.choice(self.p.lamb,self.p.k,replace=False)]
        return chosen[np.argmin(self.fitness(chosen))]

    @njit
    def basicrossover(self,indv1, indv2):
        solution = np.zeros(indv1.size, dtype=int)
        min = np.random.randint(0, indv1.size - 1)
        max = np.random.randint(0, indv1.size - 1)
        if min > max:
            min, max = max, min
        solution[min:max] = indv1.order[min:max]
        # i in alles wat er buiten zit
        for i in np.concatenate([np.arange(0, min), np.arange(max, indv1.size)]):
            candidate = indv2.order[i]
            while candidate in indv1.order[min:max]:
                candidate = indv2.order[np.where(indv1.order == candidate)[0][0]]
            solution[i] = candidate
        return solution

    # swap mutation with chance alpha of individual
    @njit
    def mutation(self, individual):
        if np.random.random() < individual.alpha:
            self.InversionMutation(individual)
            self.InversionMutation(individual)
        
    @njit
    def InversionMutation(self, individual):
        i = random.randint(0, individual.size - 1)
        j = random.randint(0, individual.size - 1)
        if i > j:
            i, j = j, i
        # reverse the order of the cities between i and j
        individual.order[i:j] = individual.order[i:j][::-1]
    
    @njit
    def full2Opt(self, individual):
        best_indiv = individual
        current_best = self.fitness(individual)
        for i in range(individual.size-1):
            for j in range(i,individual.size-1):
                new_order = individual.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_fitness = self.fitness(new_order)
                if new_fitness < current_best:
                    best_indiv = new_order
        return best_indiv
    
    @njit
    def lsoSwap(self, individual, n=1):
        best_indiv = individual
        current_best = self.fitness(individual)
        for i in range(individual.size-(n+1)):
            for j in range(1,n):
                new_order = individual.copy()
                new_order[i],new_order[i+j] = new_order[i+j],new_order[i]
                new_fitness = self.fitness(new_order)
                if new_fitness < current_best:
                    best_indiv = new_order
        return best_indiv
            

    @njit
    def distance(self,indv1,indv2):
        edges1 = np.stack((indv1,np.roll(indv1,-1)),axis=1)
        edges2 = np.stack((indv2,np.roll(indv2,-1)),axis=1)
        similarity = len(set(edges1).intersection(edges2))
        return indv1.size - similarity


    def sharedFitnessWrapper(self, X, pop=None, betaInit = 0):
        if pop is None:
            return np.vectorize(self.fitness)(X)
        
        modFitnesses = np.zeros(X.shape[0])
        max_distance = self.tsp.nCities
        alpha = 1
        sigma = 0.5 * max_distance
        for i,x in enumerate(X):
            distances = np.array([self.distance(x,y) for y in pop])
            #distances = np.vectorize(self.distance)(x,pop)
            onePlusBeta = betaInit
            within_sigma = distances <= sigma
            onePlusBeta = betaInit + np.sum(1 - (distances[within_sigma] / sigma) ** alpha)
            fitnessval = self.fitness(x)
            modFitnesses[i] = fitnessval *onePlusBeta** np.sign(fitnessval)
        
        return modFitnesses

    def sharedElimination(self):
        combined = np.concatenate((self.population, self.offspring), axis=0)
        survivors = np.zeros((self.p.lamb),dtype=Individual)
        for i in range(self.p.lamb):
            if i == 0:
                survivors[i] = combined[np.argmin(np.vectorize(self.fitness)(combined))]
            else :
                fvals = self.sharedFitnessWrapper(combined, survivors[0:i],betaInit=1)   
                idx = np.argmin(fvals)
                survivors[i] = combined[idx]
        return survivors

	# mu , lambda elimination
    def elimination(self):
        # combine the offspring and the original population
        combined = np.concatenate((self.population, self.offspring), axis=0)
        # sort the combined population by fitness
        fvals = np.vectorize(self.fitness)(combined)
        sorted_combined = combined[np.argsort(fvals)]
        return sorted_combined[:self.p.lamb]
    

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        distanceMatrix = None
        with open(filename) as file:
            distanceMatrix = np.loadtxt(file, delimiter=",")


        ### Declaration of variables ###
        self.length = len(distanceMatrix)
        self.iter = self.p.num_iters
        self.distanceMatrix = distanceMatrix
        # self.pool =  multiprocessing.Pool(multiprocessing.cpu_count())

        ### Population initialization ###
        start = timer()
        population = self.initializePopulation()
        end = timer()
        self.timings["initialization"] = end - start

        
        while(self.convergenceTest()):
            
            ### Create offspring population
            start = timer()
            self.offspring = np.zeros(p.mu, dtype=Individual)
            for i in range(self.p.mu):
                p1 = self.selection(population)
                p2 = self.selection(population)
                self.offspring[i] = self.basicrossover(p1,p2)
                self.mutation(self.offspring[i])
                # local search operator
                if self.iter % 2 == 0 and self.iter <= self.p.num_iters - 100:
                    self.offspring[i] = self.lsoSwap(self.offspring[i])
            end = timer()
            self.timings["create offspring"] = end - start
            
            ### Mutate the original population
            start = timer()
            for ind in self.population:
                self.mutation(ind)
            end = timer()
            self.timings["mutation"] = end - start
            
            # if self.iter >= self.p.num_iters - 10:
            #     self.population = self.sharedElimination()
            # else:
            #     self.population = self.elimination()
            start = timer()
            if self.iter % 50 == 0 and self.iter >= self.p.num_iters - 150:
                self.population = self.sharedElimination()
            else:
                self.population = self.elimination()
            end = timer()
            self.timings["elimination"] = end - start

            fitnesses = np.vectorize(self.fitness)(self.population)
            meanObjective = np.mean(fitnesses)
            bestObjective = np.min(fitnesses)
            bestSolution = self.population[np.argmin(fitnesses)].order
            diversityScore = len(set(fitnesses))

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            print("Mean objective: ", round(meanObjective,2),"     Best Objective: ", round(bestObjective,2),"     Iterations to go: ",self.iter, "     Diversity score: ", diversityScore, "     Time left: ", round(timeLeft,2))
            wandb.log({"mean objective": meanObjective, "best objective": bestObjective, "diversity score": diversityScore})
            wandb.log(self.timings)

            if timeLeft < 0:
                break

        # Your code here.
        return bestObjective


class Individual:
    def __init__(self, tsp, alpha, order=None):
        self.alpha = max(alpha, alpha + 0.2 * random.random())
        self.order = np.random.permutation(range(tsp.nCities))
        self.size = len(self.order)
        self.originalFitness = None
        if order is not None:
            self.order = order
            self.size = len(self.order)
        self.edges = None
    
    @jit
    def get_edges(self):
        if self.edges is None:
            self.edges = list(zip(self.order,np.roll(self.order,-1)))
        return self.edges


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
    profiler = cProfile.Profile()
    profiler.enable()
    p = Parameters(lamb=100, mu=100,num_iters=500,k=5,alpha=0.2)
    wandb.init(project="GAEC",
               config={"lamb": p.lamb, "mu": p.mu, "num_iters": p.num_iters, "k": p.k, "alpha": p.alpha})
    
    algo = r0123456(p)
    best = algo.optimize("Data/tour200.csv")

    profiler.disable()
    profiler.dump_stats("profile_2.prof")

    stats = pstats.Stats("profile.prof")

    stats.sort_stats('cumulative').print_stats()

    

    
    

