import pandas as pd
import matplotlib.pyplot as plt
import wandb
import Reporter
import numpy as np
import random
import timeit
import multiprocessing
from timeit import default_timer as timer
from numba import jit, njit
import cProfile
import pstats

@njit
def standardise(individual):
    # given a permutation, standardise it, i.e. make it start with 0
    idx = np.where(individual == 0)[0][0]
    return np.roll(individual,-idx)

@njit
def fitness(distanceMatrix,individual):
    # calculate the fitness of the individual
    # the fitness is the total distance of the cycle
    indices = np.stack((individual,np.roll(individual,-1)),axis=1)

    # return np.sum(distanceMatrix[indices[:,0],indices[:,1]])
    distance = 0
    for i in indices:
        if distanceMatrix[i[0],i[1]] == np.inf:
            return np.inf
        distance += distanceMatrix[i[0],i[1]]
    return distance

@njit
def distance(indv1,indv2):
    edges1 = np.column_stack((indv1, np.roll(indv1, -1)))
    edges2 = np.column_stack((indv2, np.roll(indv2, -1)))
    #similarity = np.count_nonzero((edges1[:,None] == edges2).all(-1).any(-1))
    similarity = 0
    for i in edges1:
        for j in edges2:
            if i[0] == j[0] and i[1] == j[1]:
                similarity += 1
    distance = indv1.size - similarity
    return distance
@njit
def full2Opt(distanceMatrix,individual):
        best_indiv = individual
        current_best = fitness(distanceMatrix,individual)
        for i in range(individual.size-1):
            for j in range(i,individual.size-1):
                new_order = individual.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_fitness = fitness(distanceMatrix,new_order)
                if new_fitness < current_best:
                    best_indiv = new_order
        return best_indiv
        
# Modify the class name to match your student number.
class r0123456:
    def __init__(self,p):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.iter = 0
        self.population = None
        self.p = p
        self.timings = {}
        self.used_dist_map = 0
    
    def initializePopulation(self):
        # create the initial population with random np.permutations
        population = np.zeros((self.p.lamb, self.length), dtype=int)
        for i in range(self.p.lamb):
            population[i] = np.random.permutation(self.length)
        return population
    

    def convergenceTest(self):
        self.iter -= 1
        return self.iter > 0
    
    
    def selection(self,population):
        # k-tournament selection
        chosen = population[np.random.choice(self.p.lamb,self.p.k,replace=False)]
        fvals = np.array([fitness(self.distanceMatrix,ind) for ind in chosen])
        return chosen[np.argmin(fvals)]

    def basicrossover(self,indv1, indv2):
        solution = np.zeros(indv1.size, dtype=int)
        min = np.random.randint(0, indv1.size - 1)
        max = np.random.randint(0, indv1.size - 1)
        if min > max:
            min, max = max, min
        solution[min:max] = indv1[min:max]
        # i in alles wat er buiten zit
        for i in np.concatenate([np.arange(0, min), np.arange(max, indv1.size)]):
            candidate = indv2[i]
            while candidate in indv1[min:max]:
                candidate = indv2[np.where(indv1 == candidate)[0][0]]
            solution[i] = candidate
        return solution

    # swap mutation with chance alpha of individual
    def mutation(self, individual, alpha):
        if np.random.random() < alpha:
            self.InversionMutation(individual)
            self.InversionMutation(individual)
        
    def InversionMutation(self, individual):
        i = random.randint(0, individual.size - 1)
        j = random.randint(0, individual.size - 1)
        if i > j:
            i, j = j, i
        # reverse the order of the cities between i and j
        individual[i:j] = individual[i:j][::-1]
    
    def full2Opt(self, individual):
        best_indiv = individual
        current_best = fitness(self.distanceMatrix,individual)
        for i in range(individual.size-1):
            for j in range(i,individual.size-1):
                new_order = individual.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_fitness = fitness(self.distanceMatrix,new_order)
                if new_fitness < current_best:
                    best_indiv = new_order
        return best_indiv
    
    def lsoSwap(self, individual, n=1):
        best_indiv = individual
        current_best = fitness(self.distanceMatrix,individual)
        for i in range(individual.size-(n+1)):
            for j in range(1,n):
                new_order = individual.copy()
                new_order[i],new_order[i+j] = new_order[i+j],new_order[i]
                new_fitness = fitness(self.distanceMatrix,new_order)
                if new_fitness < current_best:
                    best_indiv = new_order
        return best_indiv
        

    def sharedFitnessWrapper(self, X, pop=None, betaInit = 0):
        if pop is None:
            return np.array([fitness(self.distanceMatrix,x) for x in X])
        
        modFitnesses = np.zeros(X.shape[0])

        max_distance = self.length
        alpha = 0.5
        sigma = 0.3 * max_distance
        for i,x in enumerate(X):
            # distances = self.distance(x,pop)
            #distances = np.vectorize(lambda y: self.distance(x,y))(pop)
            distances = np.zeros(pop.shape[0])
            for j,y in enumerate(pop):
                dist1 = self.distanceMap.get((standardise(x).tobytes(),standardise(y).tobytes()))
                if dist1 is not None:
                    distances[j] = dist1
                    continue
                dist2 = self.distanceMap.get((standardise(x).tobytes(),standardise(y).tobytes()))
                if dist2 is not None:
                    distances[j] = dist2
                    continue
                else:
                    dist = distance(x,y)
                    self.distanceMap[(standardise(x).tobytes(),standardise(y).tobytes())] = dist
            
            onePlusBeta = betaInit
            within_sigma = distances <= sigma
            onePlusBeta = betaInit + np.sum(1 - (distances[within_sigma] / sigma) ** alpha)
            fitnessval = fitness(self.distanceMatrix,x)
            if fitnessval == np.inf:
                modFitnesses[i] = np.inf
            else:
                modFitnesses[i] = fitnessval *onePlusBeta** np.sign(fitnessval)
        
        return modFitnesses

    def sharedElimination(self, population, offspring):
        combined = np.concatenate((population,offspring), axis=0)
        survivors = np.zeros((self.p.lamb,self.length),dtype=int)
        for i in range(self.p.lamb):
            #print("survivor chosen: ", i)
            if i == 0:
                best_idx = np.argmin(np.array([fitness(self.distanceMatrix,ind) for ind in combined]))
                survivors[i] = combined[best_idx]
                np.delete(combined,best_idx,0)
            else :
                # instead of calculating the updated fitness for all, only do it for the top 50% of the population
                fvals = self.sharedFitnessWrapper(combined, survivors[0:i,:],betaInit=1)   
                idx = np.argmin(fvals)
                survivors[i] = combined[idx]
                np.delete(combined,idx,0)
        return survivors

	# mu , lambda elimination
    def elimination(self, population, offspring):
        # combine the offspring and the original population
        combined = np.concatenate((population, offspring), axis=0)
        # sort the combined population by fitness
        fvals = np.array([fitness(self.distanceMatrix,ind) for ind in combined])
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
        self.distanceMap = {}
        self.fitnessMap = {}
        # self.pool =  multiprocessing.Pool(multiprocessing.cpu_count())

        ### Population initialization ###
        start = timer()
        population = self.initializePopulation()
        alphas = np.full((self.p.lamb),max(self.p.alpha, self.p.alpha + 0.2 * np.random.random()))
        end = timer()
        self.timings["initialization"] = end - start

        
        while(self.convergenceTest()):
            
            ### Create offspring population
            start = timer()
            offspring = np.zeros((p.mu,self.length), dtype=int)
            alphas_offspring = np.full((self.p.mu),max(self.p.alpha, self.p.alpha + 0.2 * np.random.random()))
            for i in range(self.p.mu):
                p1 = self.selection(population)
                p2 = self.selection(population)
                offspring[i] = self.basicrossover(p1,p2)
                self.mutation(offspring[i],alphas_offspring[i])
                if self.iter % 5 == 0:
                    offspring[i] = full2Opt(self.distanceMatrix,offspring[i])
            end = timer()
            self.timings["create offspring"] = end - start
            
            
            ### Mutate the original population
            start = timer()
            for i,ind in enumerate(population):
                self.mutation(ind,alphas[i])
            end = timer()
            self.timings["mutation"] = end - start
            
            
            start = timer()
            if self.iter % 5 == 0 and self.iter >= 420:
                population = self.sharedElimination(population, offspring)
            else:
                population = self.elimination(population,offspring)
            end = timer()
            self.timings["elimination"] = end - start

            fitnesses = np.array([fitness(self.distanceMatrix,i) for i in population])
            
            meanObjective = np.mean(fitnesses)
            bestObjective = np.min(fitnesses)
            bestSolution = population[np.argmin(fitnesses)]
            diversityScore = len(set(fitnesses))

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            print("Mean objective: ", round(meanObjective,2),"     Best Objective: ", round(bestObjective,2),"     Iterations to go: ",self.iter, "     Diversity score: ", diversityScore, "     Time left: ", round(timeLeft,2))
            if self.p.wandb:
                wandb.log({"mean objective": meanObjective, "best objective": bestObjective, "diversity score": diversityScore} | 
                          self.timings |
                          {"used dist map": self.used_dist_map})

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
    
    def get_edges(self):
        if self.edges is None:
            self.edges = list(zip(self.order,np.roll(self.order,-1)))
        return self.edges


class TSPProblem:
    def __init__(self, distanceMatrix):
        self.distanceMatrix = distanceMatrix
        self.nCities = len(distanceMatrix)


class Parameters:
    def __init__(self, lamb, mu, num_iters, k,alpha,alpha_sharing,sigmaPerc):
        self.lamb = lamb  # Population size
        self.mu = mu  # Amount of offspring
        self.num_iters = num_iters  # How many times to create offspring
        self.k = k  # k for k-tournament selection
        self.alpha = alpha # chance of mutation
        self.wandb = True
        self.alpha_sharing = alpha_sharing
        self.sigmaPerc = sigmaPerc


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    p = Parameters(lamb=100, mu=100,num_iters=500,k=5,alpha=0.2,alpha_sharing=0.5,sigmaPerc=0.3)
    if p.wandb:
        wandb.init(project="GAEC",
                config={"lamb": p.lamb, "mu": p.mu, "num_iters": p.num_iters, "k": p.k, "alpha": p.alpha})
    
    ind = np.array([0,1,2,3,4,5,6,7,8,9])
    timeit.timeit()

    algo = r0123456(p)
    best = algo.optimize("Data/tour50.csv")

    profiler.disable()
    profiler.dump_stats("profile_2.prof")

    stats = pstats.Stats("profile_2.prof")

    stats.sort_stats('cumulative').print_stats(20)

    

    
    

