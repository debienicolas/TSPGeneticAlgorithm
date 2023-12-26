from ast import Param
from asyncio import futures
import copy
from hmac import new
from math import dist
from operator import mod
from turtle import st
from cv2 import add, exp
import pandas as pd
import matplotlib.pyplot as plt
from regex import D
from sympy import rem
import wandb
import Reporter
import numpy as np
import time
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

""" Takes as input an individual and returns a standardized individual that starts its permutation with 0."""
@njit
def standardise(individual):
    # given a permutation, standardise it, i.e. make it start with 0
    idx = 0
    for i in range(len(individual)):
        if individual[i] == 0:
            idx = i
            break
    return np.roll(individual,-idx)


@njit
def fitness(distanceMatrix,individual):

    # check if the individual is in the fitness map
    

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
def initialize_legally(distance_matrix):
    num_cities = distance_matrix.shape[0]
    permutation = np.empty(num_cities, dtype=np.int64)
    available_cities = np.ones(num_cities, dtype=np.int64)  # 1 means available

    # Start with a random city
    current_city = np.random.randint(num_cities)
    permutation[0] = current_city
    available_cities[current_city] = 0  # Mark as visited

    for i in range(1, num_cities):
        # Find cities that are not yet in the permutation and without an infinite distance
        valid_next_cities = [city for city in range(num_cities)
                             if available_cities[city] == 1 and distance_matrix[current_city, city] != np.inf]

        if len(valid_next_cities) == 0:
            # Restart if no valid next cities
            return initialize_legally(distance_matrix)

        # Choose a random next city from the valid options
        next_city = valid_next_cities[np.random.randint(len(valid_next_cities))]
        permutation[i] = next_city
        available_cities[next_city] = 0  # Mark as visited
        current_city = next_city

    return permutation


def initialize_greedily(distanceMatrix: np.ndarray):

    length = (distanceMatrix.shape)[0]
    order = np.negative(np.ones((length), dtype=int),dtype=int)
    city = np.random.randint(0, length - 1)
    order[0] = city
    i = 1
    while i < length:
        possibilities = set(range(length)) - set([elem for elem in order if elem >= 0])
        possibilities_legal = []

        for pos in possibilities:
            distance = distanceMatrix[city][pos]
            if distance < np.inf:
                possibilities_legal.append(pos)
        if not possibilities_legal:
            break
        city = min(possibilities_legal, key=lambda x: distanceMatrix[order[i - 1]][x])
        order[i] = city
        i += 1
    return order



def initializePopulation(distanceMatrix, lamb, percent_greedy=0.1):
    population = np.zeros((lamb, len(distanceMatrix)), dtype=int)
    greedy = int(lamb * percent_greedy)
    random_legal = lamb - greedy 
    for i in range(random_legal):
        population[i] = initialize_legally(distanceMatrix)
    for i in range(greedy):
        population[i + random_legal] = initialize_greedily(distanceMatrix)
    return population
    

def nearest_neighbor(dist_matrix, lamb, k):
    n = len(dist_matrix)
    population = np.zeros((lamb, n),dtype=int)

    k = int(lamb*k)

    for i in range(k):
        start = np.random.randint(0,n)
        tour = [start]
        visited = set([start])
        while len(tour) < n:
            last =  tour[-1]
            next_city = None
            min_dist = INF

            for j in range(n):
                if j not in visited and 0 < dist_matrix[last,j] < min_dist:
                    next_city = j
                    min_dist = dist_matrix[last,j]
            tour.append(next_city)
            visited.add(next_city)

        population[i] = np.array(tour)
    for i in range(k,lamb):
        population[i] = np.random.permutation(n)
    return population

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
def opt2invert(ind1,i,j):
    new_order = np.empty_like(ind1)
    new_order[:i] = ind1[:i]
    new_order[i:j+1] = ind1[i:j+1][::-1]
    new_order[j+1:] = ind1[j+1:]
    return new_order

@njit
def opt2insert(ind1,i,j):
    new_order = np.empty_like(ind1)
    new_order[:i] = ind1[:i]
    new_order[i:j+1] = ind1[i+1:j+1]
    new_order[j+1] = ind1[i]
    new_order[j+2:] = ind1[j+1:]
    return new_order

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
def opt2Sample(distM,indv,percent= 0.3):
    best_indv = indv
    current_best = fitness(distM,indv)
    size = indv.size
    subsample_1 = np.random.choice(size-1,int(percent*size),replace=False)
    for i in subsample_1:
        for j in range(i+1,size):
            lengthDelta = -distM[indv[i],indv[i+1]] - distM[indv[j],indv[(j+1)]] + distM[indv[i+1],indv[j+1]] + distM[indv[i],indv[j]]
            if lengthDelta < 0:
                new_order = opt2swap(indv,i,j)
                new_fitness = fitness(distM,new_order)
                if new_fitness < current_best:
                    best_indv = new_order
                    current_best = new_fitness
    return best_indv

@njit
def opt2withInversion(distM,indv):
    best_indiv = indv
    current_best = fitness(distM,indv)
    size = indv.size
    for i in range(size - 1):
        for j in range(i+1,size):
            new_order = opt2invert(indv,i,j)
            new_fitness = fitness(distM,new_order)
            if new_fitness < current_best:
                best_indiv = new_order
                current_best = new_fitness
    return best_indiv

@njit
def opt2withInsertion(distM,indv):
    best_indiv = indv
    current_best = fitness(distM,indv)
    size = indv.size
    for i in range(size - 1):
        for j in range(i+1,size):
            lengthDelta = -distM[indv[i],indv[i+1]] - distM[indv[j],indv[(j+1)]] + distM[indv[i+1],indv[j+1]] + distM[indv[i],indv[j]]
            if lengthDelta < 0:
                new_order = opt2insert(indv,i,j)
                new_fitness = fitness(distM,new_order)
                if new_fitness < current_best:
                    best_indiv = new_order
                    current_best = new_fitness
    return best_indiv


@njit
def randomOpt2(distM,indv, percentage=0.9):
    best_indiv = indv
    current_best = fitness(distM,indv)
    size = indv.size
    subsample = np.random.choice(size-1,int(percentage*size),replace=False)
    for i in subsample:
        for j in range(i+1,size):
            lengthDelta = -distM[indv[i],indv[i+1]] - distM[indv[j],indv[(j+1)]] + distM[indv[i+1],indv[j+1]] + distM[indv[i],indv[j]]
            if lengthDelta < 0:
                new_order = opt2swap(indv,i,j)
                new_fitness = fitness(distM,new_order)
                if new_fitness < current_best:
                    best_indiv = new_order
                    current_best = new_fitness
    return best_indiv




# Mutation: Inversion, subvector is reversed
@njit
def inversionMutation(individual):
    i,j = np.sort(np.random.choice(individual.size,2,replace=False))
    individual[i:j+1] = individual[i:j+1][::-1]
# Mutation: Swap, two random cities are swapped
def swapMutation(individual):
    i = random.randint(0, individual.size - 1)
    rang = list(range(individual.size))
    rang.pop(i)
    j = random.randint(0, individual.size - 1)
    individual[i], individual[j] = individual[j], individual[i]
# Mutation: Scramble, a random subvector is shuffled
def scrambleMutation(individual):
    i = np.random.randint(0, individual.size - 1)
    rang = list(range(individual.size))
    rang.pop(i)
    j = np.random.randint(0, individual.size - 1)
    if i > j:
        i, j = j, i
    # shuffle the cities between i and j
    individual[i:j] = np.random.permutation(individual[i:j])
# Mutation: Insert, a random city is removed and reinserted at a random position

def insertMutation(individual):
    i,j = np.sort(np.random.choice(individual.size-1,2,replace=False))
    print(i,j)
    # insert the city at position j in position i
    individual[i:j+1] = np.insert(individual[i:j],i,individual[j])


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
        # neighbor = parent1[i-1]
        # edge_table[parent1city] = add_edge(edge_table[parent1city],neighbor)
    
        # add neighbors from parent2
        parent2city = parent2[i]
        neighbor = parent2[(i+1)%size]
        edge_table[parent2city] = add_edge(edge_table[parent2city],neighbor)
        # neighbor = parent2[i-1]
        # edge_table[parent2city] = add_edge(edge_table[parent2city],neighbor)
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
        # same fitness for 300 iterations
        if self.iter > 300 and len(set(self.fitnesses[-300:])) == 1:
            return False
        return True
    
    # k-tournament selection
    def selection(self,population):

        chosen = population[np.random.choice(self.p.lamb,self.p.k,replace=False)]
        fvals = np.array([fitness(self.distanceMatrix,ind) for ind in chosen])
        return chosen[np.argmin(fvals)]
    
    # k-tournament selection with fitness sharing
    def SharingSelection(self,population,p1):
        chosen_indices = np.random.choice(self.p.lamb,self.p.k,replace=False)
        chosen = population[chosen_indices]
        modifiedFitnesses = self.sharedFitnessWrapper(chosen, p1)
        best_index = chosen_indices[np.argmin(modifiedFitnesses)]
        return population[best_index]

    
    # Mutation wrapper function
    def mutation(self, individual, alpha):
        # If in wandb sweep mode, use the mutation specified in the sweep
        if self.p.wandb:
            if wandb.config.mutation == "insert" and np.random.random() < alpha:
                insertMutation(individual)
            elif wandb.config.mutation == "inversion" and np.random.random() < alpha:
                inversionMutation(individual)
            elif wandb.config.mutation == "swap" and np.random.random() < alpha:
                swapMutation(individual)
            elif wandb.config.mutation == "scramble" and np.random.random() < alpha:
                scrambleMutation(individual)

        elif np.random.random() < alpha:
            inversionMutation(individual)
    

    def recombination(self,parent1,parent2):
        # if self.timeLeft < 150:
        #     if self.iter%4 == 0:
        #         return self.edgeCrossover(parent1,parent2)
        #         ind1,_ = self.edgeCrossover(parent1,parent2)
        #         return ind1
        #     else:
        #         return basicrossover(parent1,parent2,np.zeros(parent1.size,dtype=int)
        if self.p.wandb:
            if wandb.config.recombination == "order":
                return orderCrossover(parent1,parent2,np.zeros(parent1.size,dtype=int))
            elif wandb.config.recombination == "basic":
                return basicrossover(parent1,parent2,np.zeros(parent1.size,dtype=int))
            elif wandb.config.recombination == "edge":
                return self.edgeCrossover(parent1,parent2)
            elif wandb.config.recombination == "cycle":
                idv1,idv2 = self.cycle_crossover(parent1,parent2)
                # return the best of the two offspring
                fit1 = fitness(self.distanceMatrix,idv1)
                fit2 = fitness(self.distanceMatrix,idv2)
                if fit1 < fit2:
                    return idv1
                else:
                    return idv2
                
        else:
            idv1,idv2 = self.cycle_crossover(parent1,parent2)
                # return the best of the two offspring
            fit1 = fitness(self.distanceMatrix,idv1)
            fit2 = fitness(self.distanceMatrix,idv2)
            if fit1 < fit2:
                return idv1
            else:
                return idv2
    
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

        offspring1, offspring2 = np.zeros(indv1.size,dtype=int), np.zeros(indv1.size,dtype=int)

        # Initialize the offspring with None (using NaN for floating point representation)
        offspring1[:] = -1
        offspring2[:] = -1

        # Start cycle crossover
        cycle_num = 1
        while np.any(offspring1 == -1):
            if cycle_num % 2 == 1:  # If it's an odd cycle
                start = np.where(offspring1== -1)[0][0]
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
                start = np.where(offspring1 == -1 )[0][0]
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
        

    def sharedFitnessWrapper(self,orig_fitnesses, X, pop=None, betaInit = 0):
        if pop is None:
            return np.array([fitness(self.distanceMatrix,x) for x in X])
        
        modFitnesses = np.zeros(X.shape[0])

        max_distance = self.length
        if self.p.wandb:
            alpha = wandb.config.alpha_sharing
            sigma = wandb.config.sigmaPerc * max_distance
        else:
            alpha = self.p.alpha_sharing
            sigma = self.p.sigmaPerc * max_distance

        for i,x in enumerate(X):
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
            # add to fitness hashmap
            fitnessval = orig_fitnesses[i]
                
            if fitnessval == np.inf:
                modFitnesses[i] = np.inf
            else:
                modFitnesses[i] = fitnessval *onePlusBeta** np.sign(fitnessval)
        
        return modFitnesses

    def sharedElimination(self, population, offspring):
        combined = np.concatenate((population,offspring), axis=0)
        survivors = np.zeros((self.p.lamb,self.length),dtype=int)
        fitnesses = np.array([fitness(self.distanceMatrix,ind) for ind in combined])
        for i in range(self.p.lamb):
            #print("survivor chosen: ", i)
            if i == 0:
                best_idx = np.argmin(np.array([fitness(self.distanceMatrix,ind) for ind in combined]))
                survivors[i] = combined[best_idx]
                np.delete(combined,best_idx,0)
            else :
                # instead of calculating the updated fitness for all, only do it for the top 50% of the population ? 
                # take random subset of the combined population
                fvals = self.sharedFitnessWrapper(fitnesses,combined, survivors[0:i,:],betaInit=1)
                idx = np.argmin(fvals)
                survivors[i] = combined[idx]
                np.delete(combined,idx,0)
        return survivors
    
    # def sharedElimination_k(self,population,offspring,k_elim):
    #     combined = np.concatenate((population,offspring), axis=0)
    #     fitnesses = np.empty((len(combined)))
    #     for i, ind in enumerate(combined):
    #         fitnesses[i] = self.fitnessMap.get(hash(standardise(ind).tobytes()),fitness(self.distanceMatrix,ind))
    #     survivors = np.zeros((self.p.lamb,self.length),dtype=int)
    #     best_indx = np.argmin(fitnesses)        
    #     survivors[0] = combined[best_indx]
    #     for i in range(self.p.lamb-1):
    #         fvals = self.sharedFitnessWrapper


	# mu , lambda elimination
    def elimination(self, population, offspring):
        # combine the offspring and the original population
        combined = np.concatenate((population, offspring), axis=0)
        # sort the combined population by fitness
        fvals = np.array([fitness(self.distanceMatrix,ind) for ind in combined])
        sorted_combined = combined[np.argsort(fvals)]
        return sorted_combined[:self.p.lamb]
    
    def sharedEliminDecay(self,timeLeft):
        timePassed = 300 - timeLeft
        return math.exp(-wandb.config.sharedElimDecay_alpha * (((timePassed)/300)**wandb.config.sharedElimDecay_n)) - wandb.config.sharedElimDecay_c
    def basicCrossoberDecay(self,timeLeft,alpha,n):
        return math.exp(-alpha * ((timeLeft/300)** n))
    def LSOGrowth(self,timeLeft):
        timePassed = 300 - timeLeft
        return 1 - math.exp(-wandb.config.LSO_alpha * ((timePassed/300)**wandb.config.LSO_n)) + wandb.config.LSO_c
    def edgeCrossoverGrowth(self,timeLeft,alpha,n):
        return 1 - math.exp(-alpha * ((timeLeft/300)** n))

    def lambDecay(self,timeLeft):
        timePassed = 300 - timeLeft
        lamb_alpha = -math.log(wandb.config.lambEnd/wandb.config.lambStart)
        return int(wandb.config.lambStart*math.exp(-lamb_alpha * ((timePassed/300)**wandb.config.lambDecay_n)))

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

        ### Population initialization ###
        start = timer()
        population = initializePopulation(distanceMatrix, self.p.lamb,wandb.config.percent_greedy_init)
        alphas = np.full((self.p.lamb),max(self.p.alpha, self.p.alpha + 0.2 * np.random.random()))
        # normalize the the individuals
        for i in range(self.p.lamb):
            population[i] = standardise(population[i])
        end = timer()
        self.timings["initialization"] = end - start

        self.elimdecay = self.sharedEliminDecay(300)
        self.lsogrowth = self.LSOGrowth(300)
        self.fitnesses = []
        self.timeLeft = 300
            
    
        while(self.convergenceTest()):
            self.p.mu = self.p.lamb

            
            ### Create offspring population ### 
            start = timer()
            offspring = np.zeros((self.p.mu,self.length), dtype=int)
            alphas_offspring = np.full((self.p.mu),max(self.p.alpha, self.p.alpha + 0.2 * np.random.random()))
            
            for i in range(self.p.mu):
                p1 = self.selection(population)
                p2 = self.selection(population)
                
                offspring[i] = self.recombination(p1,p2)
                
                self.mutation(offspring[i],alphas_offspring[i])
                ### apply local search operator to the offspring ###
                if self.lsogrowth > np.random.random():
                    offspring[i] = opt2Sample(self.distanceMatrix,offspring[i],percent=wandb.config.LSOPercent)

                    
                
            end = timer()
            self.timings["Create offspring + LSO"] = end - start


            ### Mutate the original population and apply local search operator ###
            start = timer()
            for i,ind in enumerate(population):
                self.mutation(ind,alphas[i])
                if self.lsogrowth > np.random.random():
                    population[i] = opt2Sample(self.distanceMatrix,ind,percent=wandb.config.LSOPercent)
                
        

            end = timer()
            self.timings["Mutation + LSO"] = end - start

            ### Shared fitness elimination ###
            start = timer()
            if self.elimdecay > np.random.random():
                population = self.sharedElimination(population,offspring)
            else:
                population = self.elimination(population,offspring)
            self.timings["Elimination"] = timer() - start

            fitnesses = np.array([fitness(self.distanceMatrix,i) for i in population])
            
            meanObjective = np.mean(fitnesses)
            bestObjective = np.min(fitnesses)
            self.fitnesses.append(bestObjective)
            bestSolution = population[np.argmin(fitnesses)]
            diversityScore = len(set(fitnesses))

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            self.timeLeft = timeLeft
            print("Mean objective: ", round(meanObjective,2),"     Best Objective: ", round(bestObjective,2),"     Iteration: ",self.iter, "     Diversity score: ", diversityScore, "     Time left: ", round(timeLeft,2))
            if self.p.wandb:
                wandb.log({"mean objective": meanObjective, "best objective": bestObjective, "diversity score": diversityScore} | 
                          self.timings |
                          {"used dist map": self.used_dist_map} |
                          {"solution": bestSolution.tolist()} |
                          {"elimdecay": self.elimdecay, "LSOgrowth": self.lsogrowth,"lambDecay":self.p.lamb} |
                          {"time left": timeLeft,"time": 300-timeLeft} )

            if timeLeft < 0:
                break
            self.lsogrowth = self.LSOGrowth(timeLeft)
            self.elimdecay = self.sharedEliminDecay(timeLeft)
            self.p.lamb = self.lambDecay(timeLeft)
            

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
        #self.sharedElim = np.arange(0,wandb.config.fitness_sharing_stop_iter,wandb.config.fitness_sharing_occurrence)
        #self.sharedElim = np.arange(0,120,1)



if __name__ == "__main__":

    
    #wandb.init()
    # tourNumber = wandb.config.tour
    #tour =  f"Data/tour{tourNumber}.csv"

    
    wandb.init(project="GAEC")
    wandb.config.recombination = "basic"
    wandb.config.mutation = "inversion"

    wandb.config.LSO_alpha = 2#2 #0.97
    wandb.config.LSO_n = 1#4
    wandb.config.LSO_c = 0.3

    wandb.config.sharedElimDecay_alpha = 1
    wandb.config.sharedElimDecay_n = 50
    wandb.config.sharedElimDecay_c = 00

    wandb.config.lambStart = 100
    wandb.config.lambEnd = 30
    wandb.config.lambDecay_n = 0.6

    wandb.config.alpha_sharing = 2.5
    wandb.config.sigmaPerc = 0.3
    wandb.config.percent_greedy_init = 0.20 #0.05

    wandb.config.lamb = wandb.config.lambStart #40
    wandb.config.mu = wandb.config.lambStart #40
    wandb.config.k = 7
    wandb.config.alpha = 0.1

    wandb.config.LSOPercent = 0.9
    
    wandb.config.tour = 750
    tour =  f"Data/tour{wandb.config.tour}.csv"
    
    
    # p = Parameters(lamb=15, mu=15,num_iters=1500,k=9,alpha=0.05,alpha_sharing=0.20,sigmaPerc=0.4)
    p = Parameters(lamb=wandb.config.lambStart, mu=wandb.config.lambStart,num_iters=5000,k=wandb.config.k,alpha=wandb.config.alpha,alpha_sharing=wandb.config.alpha_sharing,sigmaPerc=wandb.config.sigmaPerc)
    
    # if p.wandb:
    #     wandb.init(project="GAEC", config={"lamb": p.lamb, "mu": p.mu, "num_iters": p.num_iters, "k": p.k, "alpha": p.alpha, "tour":tourNumber, "alpha_sharing": p.alpha_sharing, "sigmaPerc": p.sigmaPerc})
    
      
    # profiler = cProfile.Profile()
    # profiler.enable()
    algo = r0123456(p)
    best = algo.optimize(tour)
    # profiler.disable()
    # profiler.dump_stats("profile.prof")
    # stats = pstats.Stats("profile.prof")
    # stats.sort_stats('cumulative').print_stats(20)

    

    
    

