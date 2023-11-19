from os import replace
import Reporter
import numpy as np
import random
import time


# Modify the class name to match your student number.


class Parameters:
    def __init__(self, lamb, mu, num_iters, k, alpha):
        self.lamb = lamb
        self.mu = mu
        self.num_iters = num_iters
        self.k = k
        self.alpha = alpha


class TSPProblem:
    def __init__(self, distanceMatrix):
        self.distanceMatrix = distanceMatrix
        self.nCities = len(distanceMatrix)

class r0810938:
    def __init__(self,p):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.p = p

    def initPopulation(self):
        self.population = np.array([np.random.permutation(self.tsp.nCities) for _ in range(self.p.lamb)])
        self.alphas = np.array([max(self.p.alpha, self.p.alpha + 0.1 * random.random()) for _ in range(self.p.lamb)])
    
    def convergenceTest(self):
         self.iter -= 1
         return self.iter > 0

    # moet een indivdual meegeven
    def fitness(self, individual):
        distance = []
        distance = 0
        for i in range(self.tsp.nCities-1):
            distance += self.tsp.distanceMatrix[individual[i]][individual[i+1]]
        distance += self.tsp.distanceMatrix[individual[self.tsp.nCities - 1]][individual[0]]
        
        return distance

    ### K tournament selection
    def selection(self):
        candidates = np.random.choice(np.array(range(self.tsp.nCities)),self.p.k, replace=False)
        return candidates[np.argmin([self.fitness(self.population[c]) for c in candidates])]
    
    ### Recombination
    def recombination(self,p1,p2):
         return self.basiccrossover(p1,p2)

    ### Basic crossover
    def basiccrossover(self,indv1, indv2):
        indv1 = self.population[indv1]
        indv2 = self.population[indv2]
        solution = np.array([None] * self.tsp.nCities)
        min = np.random.randint(0, self.tsp.nCities- 1)
        max = np.random.randint(0, self.tsp.nCities - 1)
        if min > max:
            min, max = max, min
        solution[min:max] = indv1[min:max]
        # i in alles wat er buiten zit
        for i in np.concatenate([np.arange(0, min), np.arange(max, self.tsp.nCities)]):
            candidate = indv2[i]
            while candidate in indv1[min:max]:
                candidate = indv2[np.where(indv1 == candidate)[0][0]]
            solution[i] = candidate

        return solution

    ### Mutation
    def mutation(self,i,offspring):
        self.inversion(i,offspring)
        self.inversion(i,offspring)

    ### Mutation: swap
    def swap(self,i):
        indv = self.population[i]
        if random.random() < self.alphas[i]:
            i = random.randint(0,indv.size-1)
            j = random.randint(0,indv.size-1)
            indv.order[i], indv.order[j] = indv.order[j], indv.order[i]
    
    ### Mutation: inversion
    def inversion(self,i,offspring):
        indv = self.offspring[i] if offspring else self.population[i]
        if random.random() < self.p.alpha:
            i = random.randint(0,self.tsp.nCities-1)
            j = random.randint(0,self.tsp.nCities-1)
            if i > j:
                i,j = j,i
            indv[i:j] = indv[i:j][::-1]
        if offspring:
            self.offspring[i] = indv
        else:
            self.population[i] = indv
    
    ### Elimination (mu,lambda)
    def elimination(self):
        offspring_indices = np.argsort(np.apply_along_axis(self.fitness,1,self.offspring))
        sorted_offspring = self.offspring[offspring_indices]
        return sorted_offspring[:self.p.lamb]
    
    # The evolutionary algorithm’s main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        distanceMatrix = None  	
        with open(filename) as file:
            distanceMatrix = np.loadtxt(file, delimiter=",")
        
        # Create TSP problem instance and set parameters
        self.tsp = TSPProblem(distanceMatrix)
        self.iter = self.p.num_iters

        # Initialise population
        self.initPopulation()
        
        while(self.convergenceTest()):
            
            start = time.time()

            ### create offspring
            self.offspring = np.zeros((self.p.mu,self.tsp.nCities),dtype=int)
            for i in range(self.p.mu):
                p1 = self.selection()
                p2 = self.selection()
                self.offspring[i] = self.recombination(p1,p2)
                self.mutation(i, offspring=True)

            ### Mutate the original population
            for ind in range(len(self.population)):
                self.mutation(ind,offspring=False)
            
            ### Elimination
            print("elimination")
            self.population = self.elimination()
            
            
            fitnesses  = np.array([self.fitness(i) for i in self.population])

            stop = time.time()

            meanObjective = np.mean(fitnesses)
            bestObjective = np.min(fitnesses)
            bestSolution = self.population[np.argmin(fitnesses)]

            print("iteration: ",self.p.num_iters - self.iter, "    Mean Objective: ", 
                  meanObjective,"   Best Objective: ", bestObjective, "    Time: ", stop-start)

            timeLeft = self.reporter.report(meanObjective , bestObjective , bestSolution)
            if timeLeft < 0:
                break
    
        # Your code here.
        return 0

if __name__ == '__main__':
    p = Parameters(lamb=200,mu=800,num_iters=500,k=5,alpha=0.25)
    algo = r0810938(p)
    algo.optimize("Data/tour50.csv")



