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

# Modify the class name to match your student number.
class r0123456:
    def __init__(self,p):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.iter = 0
        self.population = None
        self.p = p
        self.timings = {}
    

    def fitness(self,individual):
        # calculate the fitness of the individual
        # the fitness is the total distance of the cycle
        distance = 0
        for i in range(individual.size - 1):
            distance += self.tsp.distanceMatrix[individual.order[i]][individual.order[i + 1]]
        # remember to add the distance from the last city to the first city
        distance += self.tsp.distanceMatrix[individual.order[individual.size - 1]][
            individual.order[0]
        ]
        return distance


    # k tournament selection
    
    def selection(self):
        candidates = np.random.choice(self.population, self.p.k)
        # candidates = random.sample(self.population, self.p.k)
        return candidates[np.argmin([self.fitness(c) for c in candidates])]

    def initializePopulation(self):
        self.population = np.array([Individual(self.tsp,alpha=self.p.alpha,order=None) for _ in range(self.p.lamb)],dtype=Individual)
        for indv in self.population:
            indv.originalFitness = self.fitness(indv)
        
        # while any(np.isinf([self.fitness(individual) for individual in self.population])):
        #     for i, individual in enumerate(self.population):
        #         if np.isinf(self.fitness(individual)):
        #             self.population[i] = Individual(self.tsp,alpha=self.p.alpha,order=None)
        
        # return [Individual(tsp) for _ in range(p.lamb)]

    def convergenceTest(self):
        self.iter -= 1
        return self.iter > 0

    def basicrossover(self,indv1, indv2):
        solution = np.array([None] * indv1.size)
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

        return Individual(self.tsp, alpha= self.p.alpha, order=solution)

    def recombination(self,indv1, indv2):
        return self.basicrossover(indv1, indv2)

    # swap mutation with chance alpha of individual
    def mutation(self, individual):
        if random.random() < individual.alpha:
            self.InversionMutation(individual)
            self.InversionMutation(individual)
        
    
    def InversionMutation(self, individual):
        i = random.randint(0, individual.size - 1)
        j = random.randint(0, individual.size - 1)
        if i > j:
            i, j = j, i
        # reverse the order of the cities between i and j
        individual.order[i:j] = individual.order[i:j][::-1]
    
    def twoOpt(self, individual):
        max_samples = self.tsp.nCities
        samples = int(max_samples * 0.2)
        best_indiv = individual
        current_best = self.fitness(individual)
        for i in np.random.choice(range(individual.size-1), samples):
            for j in np.random.choice(range(individual.size-samples), samples):
                new_order = individual.order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_individual = Individual(self.tsp, alpha=self.p.alpha, order=new_order)
                new_fitness = self.fitness(new_individual)
                if new_fitness < current_best:
                    best_indiv = new_individual
        return best_indiv
    
    def lsoSwap(self, individual):
        best_indiv = individual
        current_best = self.fitness(individual)
        for i in range(individual.size-3):
            for j in range(2):
                new_order = individual.order.copy()
                new_order[i],new_order[i+j] = new_order[i+j],new_order[i]
                new_individual = Individual(self.tsp, alpha=self.p.alpha, order=new_order)
                new_fitness = self.fitness(new_individual)
                if new_fitness < current_best:
                    best_indiv = new_individual
        return best_indiv
            

    
    def distance(self,indv1,indv2):
        if not isinstance(indv2,Individual):
            return np.inf
        edges1 = indv1.get_edges()
        edges2 = indv2.get_edges()
        similarity = np.sum(np.isin(edges1,edges2))
        return indv1.size - similarity

    def sharedFitnessWrapper(self, X, pop=None,betaInit = 0):
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
    
    def createOffspring_individual(self, i):
        p1 = self.selection()
        p2 = self.selection()
        self.offspring[i] = self.recombination(p1,p2)
        self.mutation(self.offspring[i])
        # local search operator
        if self.iter % 2 == 0 and self.iter <= self.p.num_iters - 100:
            self.offspring[i] = self.lsoSwap(self.offspring[i])
        return self.offspring[i]
    

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        distanceMatrix = None
        with open(filename) as file:
            distanceMatrix = np.loadtxt(file, delimiter=",")
        # Your code here.

        # Create a TSP problem instance.
        self.tsp = TSPProblem(distanceMatrix)

        ### Population initialization
       
        self.iter = p.num_iters
        start = timer()
        self.initializePopulation()
        end = timer()
        self.timings["initialization"] = end - start

        pool = multiprocessing.Pool()


        while(self.convergenceTest()):
            
            ### Create offspring population
            start = timer()
            self.offspring = np.zeros(p.mu, dtype=Individual)
            if self.iter % 2 == 0 and self.iter <= self.p.num_iters - 100:
                offspring_indv = pool.map(self.createOffspring_individual, range(p.mu))
                self.offspring = np.array(offspring_indv)
            else:
                for i in range(self.p.mu):
                    self.offspring[i] = self.createOffspring_individual(i)
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
    p = Parameters(lamb=100, mu=100,num_iters=500,k=5,alpha=0.2)
    wandb.init(project="GAEC",
               config={"lamb": p.lamb, "mu": p.mu, "num_iters": p.num_iters, "k": p.k, "alpha": p.alpha})
    
    algo = r0123456(p)
    best = algo.optimize("Data/tour200.csv")
    

    
    

