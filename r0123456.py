from asyncio import futures
import copy
from hmac import new
from math import dist
from turtle import st
from cv2 import add
import pandas as pd
import matplotlib.pyplot as plt
from sympy import rem
import wandb
import Reporter
import numpy as np
import random
import timeit
from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer
from numba import jit, njit
import cProfile
import pstats
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

INF = 1000000000000000

@njit
def standardise(individual):
    # given a permutation, standardise it, i.e. make it start with 0
    idx = 0
    for i in range(len(individual)):
        if individual[i] == 0:
            idx = i
            break
    return np.roll(individual,-idx)
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
        if distanceMatrix[i[0],i[1]] >= INF:
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
def opt2swap(indv1, i,j):
    new_order = np.empty_like(indv1)
    # take route[0] to route[i] and add them in order to new_route
    new_order[:i+1] = indv1[:i+1]
    # take route[i+1] to route[j] and add them in reverse order to new_route
    new_order[i+1:j+1] = indv1[j:i:-1]
    # take route[j+1] to end and add them in order to new_route
    new_order[j+1:] = indv1[j+1:]
    return new_order

@njit
def opt2Full(distM,offspring):
    for k in range(offspring.size):
        offspring[k] = opt2(distM,offspring[k])
    return offspring

@njit 
def opt2(distM,indv):
    best_indiv = indv
    current_best = fitness(distM,indv)
    size = indv.size
    for i in range(size - 1):
        for j in range(i+1,size):
            lengthDelta = -distM[indv[i],indv[i+1]] - distM[indv[j],indv[(j+1)]] + distM[indv[i+1],indv[j+1]] + distM[indv[i],indv[j]]
            if lengthDelta < 0:
                new_order = opt2swap(indv,i,j)
                new_fitness = fitness(distM,new_order)
                if new_fitness < current_best:
                    best_indiv = new_order
                    current_best = new_fitness
    return best_indiv


@njit
def inversionMutation(individual):
    i,j = np.sort(np.random.choice(individual.size,2,replace=False))
    individual[i:j+1] = individual[i:j+1][::-1]


def add_edge(s,n):
    if n in s:
        s.remove(n)
        s.add(-n)
    else:
        s.add(n)
    return s
    

def construct_edge_table(parent1, parent2):
    size = parent1.size
    edge_table = {i:set() for i in range(size)}
    for i in range(size):
        # add neighbors from parent1
        neighbor = parent1[(i+1)%size]
        parent1city = parent1[i]
        edge_table[parent1city] = add_edge(edge_table[parent1city],neighbor)
        neighbor = parent1[i-1]
        edge_table[parent1city] = add_edge(edge_table[parent1city],neighbor)
    
        # add neighbors from parent2
        parent2city = parent2[i]
        neighbor = parent2[(i+1)%size]
        edge_table[parent2city] = add_edge(edge_table[parent2city],neighbor)
        neighbor = parent2[i-1]
        edge_table[parent2city] = add_edge(edge_table[parent2city],neighbor)
    return edge_table


@njit
def orderCrossover(indv1, indv2,offspring):
        i,j = np.sort(np.random.choice(indv1.size,2,replace=False))
        offspring[i:j+1] = indv1[i:j+1]

        for k in range(indv1.size):
            if indv2[(j+1+k)%indv1.size] not in offspring:
                for l in range(offspring.size):
                    if offspring[(l+1+j)%indv1.size] == 0:
                        offspring[(l+1+j)%indv1.size] = indv2[(j+1+k)%indv1.size]
                        break
        return offspring
@njit
def find_alternate_candidate(indv1,indv2,candidate):
    for i in range(indv1.size):
        if indv1[i] == candidate:
            return indv2[i]
    return None
@njit
def basicrossover(indv1, indv2,offspring):
        min,max = np.sort(np.random.choice(indv1.size,2,replace=False))
        offspring[min:max] = indv1[min:max]
        # i in alles wat er buiten zit
        for i in range(indv1.size):
            if i < min or i >= max:
                candidate = indv2[i]
                while candidate in indv1[min:max]:
                    candidate = find_alternate_candidate(indv1,indv2,candidate)
                offspring[i] = candidate
        return offspring


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
        self.iter += 1
        return self.iter <= self.p.num_iters
    
    
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
    
    def edgeCrossover(self, indv1, indv2):
        # Construct edge table
        edge_table = construct_edge_table(indv1, indv2)
        length = len(indv1)
        # Initialize new order with -1
        new_order = np.full(length, -1, dtype=int)
        
        # Pick an initial element at random and put it in the offspring
        node = random.randint(0, length - 1)	
        new_order[0] = node
        forward = True
        counter = 1

        while counter < length:
            # Remove all references to 'node' from the table
            for key in edge_table.keys():
                edge_table[key].discard(node)
                edge_table[key].discard(-node)

            # Examine set for 'node'
            current_set = edge_table[node]
            if current_set:
                # If there is a common edge, pick that to be the next node
                double_edge_node = next((elem for elem in current_set if elem < 0), None)
                if double_edge_node is not None:
                    node = -double_edge_node
                else:
                    # Otherwise, pick the entry in the set which itself has the shortest list
                    node = min(current_set, key=lambda x: len(edge_table[x]))

                new_order[counter] = node
                counter += 1
            else:
                # In case of reaching an empty set, examine the other end of the offspring
                if forward:
                    forward = False
                    new_order[:counter] = new_order[:counter][::-1]
                    node = new_order[counter - 1]
                else:
                    # Otherwise, choose a new element at random
                    forward = True
                    new_order[:counter] = new_order[:counter][::-1]
                    remaining = set(range(length)) - set(new_order)
                    node = random.choice(list(remaining))
                    new_order[counter] = node
                    counter += 1

        return new_order

    def orderCrossover(self, indv1, indv2):
        offspring = np.zeros(indv1.size, dtype=int)
        i,j = np.sort(np.random.choice(indv1.size,2,replace=False))
        offspring[i:j+1] = indv1[i:j+1]

        for k in range(indv1.size):
            if indv2[(j+1+k)%indv1.size] not in offspring:
                for l in range(offspring.size):
                    if offspring[(l+1+j)%indv1.size] == 0:
                        offspring[(l+1+j)%indv1.size] = indv2[(j+1+k)%indv1.size]
                        break
        return offspring

    def cycle_crossover(self, indv1, indv2):

        offspring1, offspring2 = np.zeros(indv1.size), np.zeros(indv1.size)

        # Initialize the offspring with None (using NaN for floating point representation)
        offspring1[:] = np.nan
        offspring2[:] = np.nan

        # Start cycle crossover
        cycle_num = 1
        while np.isnan(offspring1).any():
            if cycle_num % 2 == 1:  # If it's an odd cycle
                start = np.where(np.isnan(offspring1))[0][0]
                indices = [start]
                val = indv1[start]
                while True:
                    start = np.where(indv2 == val)[0][0]
                    if start in indices:
                        break
                    indices.append(start)
                    val = indv1[start]

                offspring1[indices] = indv1[indices]
                offspring2[indices] = indv2[indices]
            else:  # If it's an even cycle
                start = np.where(np.isnan(offspring1))[0][0]
                indices = [start]
                val = indv2[start]
                while True:
                    start = np.where(indv1 == val)[0][0]
                    if start in indices:
                        break
                    indices.append(start)
                    val = indv2[start]

                offspring1[indices] = indv2[indices]
                offspring2[indices] = indv1[indices]
            cycle_num += 1

        return offspring1, offspring2


    # swap mutation with chance alpha of individual
    def mutation(self, individual, alpha):
        if np.random.random() < alpha:
            inversionMutation(individual)
    
            
    def InversionMutation(self, individual):
        i,j = np.sort(np.random.choice(individual.size,2,replace=False))
        # reverse the order of the cities between i and j
        individual[i:j+1] = individual[i:j+1][::-1]
    
    def swapMutation(self, individual):
        i = random.randint(0, individual.size - 1)
        rang = list(range(individual.size))
        rang.pop(i)
        j = random.randint(0, individual.size - 1)
        individual[i], individual[j] = individual[j], individual[i]
    
    def scrambleMutation(self, individual):
        i = np.random.randint(0, individual.size - 1)
        rang = list(range(individual.size))
        rang.pop(i)
        j = np.random.randint(0, individual.size - 1)
        if i > j:
            i, j = j, i
        # shuffle the cities between i and j
        individual[i:j] = np.random.permutation(individual[i:j])
    
    def lso(self, individual):
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
                key = (hash(standardise(x).tobytes()),hash(standardise(y).tobytes()))
                dist1 = self.distanceMap.get(key,None)
                if dist1 is not None:
                    distances[j] = dist1
                    continue
                key = (hash(standardise(y).tobytes()),hash(standardise(x).tobytes()))
                dist2 = self.distanceMap.get(key,None)
                if dist2 is not None:
                    distances[j] = dist2
                    continue
                else:
                    dist = distance(x,y)
                    key = (hash(standardise(x).tobytes()),hash(standardise(y).tobytes()))
                    self.distanceMap[key] = dist
            
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
    
    def createOffspring(self, population, alpha):
        p1 = self.selection(population)
        p2 = self.selection(population)
        if self.iter != 0:
            offspring = self.edgeCrossover(p1,p2)
        else:
            offspring = basicrossover(p1,p2,np.zeros(self.length,dtype=int))
        self.mutation(offspring,alpha)
        return offspring

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        distanceMatrix = None
        with open(filename) as file:
            distanceMatrix = np.loadtxt(file, delimiter=",")


        ### Declaration of variables ###
        self.length = len(distanceMatrix)
        self.iter = 0
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
            # for i in range(self.p.mu):
            #     p1 = self.selection(population)
            #     p2 = self.selection(population)
            #     start = timer()
            #     if self.iter % 4 == 0:
            #         offspring[i] = self.edgeCrossover(p1,p2)
            #     else:
            #         offspring[i] = basicrossover(p1,p2,offspring[i])
            #     end = timer()
            #     self.timings["create offspring"] = end - start
            #     self.mutation(offspring[i],alphas_offspring[i])
            #     if self.iter % 5 == 0:
            #         start = timer()
            #         offspring[i] = opt2(self.distanceMatrix,offspring[i])
            #         end = timer()
            #         self.timings["LSO"] = end - start

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.createOffspring, population,alphas_offspring[i]) for i in range(self.p.mu)]
                for i, future in enumerate(as_completed(futures)):
                    offspring[i] = future.result()
            end = timer()
            self.timings["create offspring"] = end - start

            if self.iter % 3 == 0:
                start = timer()
                for i in range(self.p.mu):
                    offspring[i] = opt2(self.distanceMatrix,offspring[i])
                end = timer()
                self.timings["LSO"] = end - start
            

            ### Mutate the original population
            start = timer()
            for i,ind in enumerate(population):
                self.mutation(ind,alphas[i])
            end = timer()
            self.timings["mutation"] = end - start
            
            
            start = timer()
            if self.iter == self.p.sharedElim:
                population = self.sharedElimination(population, offspring)
                self.p.sharedElim = self.p.sharedElim + math.floor(math.pow(2, math.log(self.p.sharedElim+1, 3)))
                print("shared elimination")
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

            print("Mean objective: ", round(meanObjective,2),"     Best Objective: ", round(bestObjective,2),"     Iteration: ",self.iter, "     Diversity score: ", diversityScore, "     Time left: ", round(timeLeft,2))
            if self.p.wandb:
                wandb.log({"mean objective": meanObjective, "best objective": bestObjective, "diversity score": diversityScore} | 
                          self.timings |
                          {"used dist map": self.used_dist_map})

            if timeLeft < 0:
                break

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
        self.wandb = False
        self.alpha_sharing = alpha_sharing
        self.sigmaPerc = sigmaPerc
        self.sharedElim = 1


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    p = Parameters(lamb=100, mu=100,num_iters=700,k=5,alpha=0.05,alpha_sharing=0.5,sigmaPerc=0.3)
    if p.wandb:
        wandb.init(project="GAEC",
                   name = "EdgeCrossover threads(4) evo shared fitness",
                config={"lamb": p.lamb, "mu": p.mu, "num_iters": p.num_iters, "k": p.k, "alpha": p.alpha})
    
    

    algo = r0123456(p)
    best = algo.optimize("Data/tour200.csv")

    # indv1 = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])
    # indv2 = np.array([0,9, 3, 7, 8, 2, 6, 5, 1, 4])
    # print(construct_edge_table(indv1,indv2))
    # print(algo.edgeCrossover(indv1,indv2))
    # print(algo.orderCrossover(indv1,indv2))
    # print(algo.cycle_crossover(indv1,indv2))
   
    profiler.disable()
    profiler.dump_stats("profile.prof")

    stats = pstats.Stats("profile.prof")

    stats.sort_stats('cumulative').print_stats(20)

    

    
    

