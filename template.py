import Reporter
import numpy as np
import random
import time

# Modify the class name to match your student number.

class Individual:
    def __init__(self,tsp,order=None,alpha=None):
        self.alpha = max(0.5,0.5+0.2*random.random()) 
        self.order = np.random.permutation(range(tsp.nCities))
        self.size = len(self.order)

		# If you don't want individual to be random
        if order is not None:
            self.order = order
            self.size = len(self.order)
        if alpha is not None:
            self.alpha = alpha

class Parameters:
	def __init__(self, lamb, mu, num_iters,k):
		self.lamb = lamb
		self.mu = mu
		self.num_iters = num_iters
		self.k = k

class TSPProblem:
	def __init__(self,distanceMatrix):
		self.distanceMatrix = distanceMatrix
		self.nCities = len(distanceMatrix)

class r0810938:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.iter = 0
        self.population = None
        self.tsp = None
        self.p = None

    def initPopulation(self):
        return [Individual(self.tsp) for _ in range(self.p.lamb)]

         
    def nearestNeighbour(self):
        start = np.random.randint(0,self.tsp.nCities)
        path = [start]
        mask = np.ones(self.tsp.nCities,dtype=bool)

        mask[start] = False
        for i in range(self.tsp.nCities-1):
            last = path[-1]
            next_ind = np.argmin(self.tsp.distanceMatrix[last][mask])
            next_city = np.arange(self.tsp.nCities)[mask][next_ind]
            path.append(next_city)
            mask[next_city] = False
        return Individual(self.tsp,order=np.array(path))
             
    
    def convergenceTest(self):
         self.iter -= 1
         return self.iter > 0

    def fitness(self, individual):
        distance = sum([self.tsp.distanceMatrix[individual.order[i]][individual.order[i+1]] for i in range(individual.size - 1)])
        distance += self.tsp.distanceMatrix[individual.order[individual.size - 1]][individual.order[0]]
        return distance

    ### K tournament selection
    def selection(self):
        candidates = random.sample(self.population,self.p.k)
        return candidates[np.argmin([self.fitness(c) for c in candidates])]
    
    ### Recombination
    def recombination(self,p1,p2):
         return self.basiccrossover(p1,p2)

    ### Basic crossover
    def basiccrossover(self,indv1, indv2):
        solution = np.array([None]*indv1.size)
        min = np.random.randint(0,indv1.size-1)
        max = np.random.randint(0,indv1.size-1)
        min = 4
        max = 4
        if min > max:
            min,max = max,min
        solution[min:max] = indv1.order[min:max]
        # i in alles wat er buiten zit
        for i in np.concatenate([np.arange(0,min),np.arange(max,indv1.size)]):
            candidate = indv2.order[i]
            while candidate in indv1.order[min:max]:
                candidate = indv2.order[np.where(indv1.order == candidate)[0][0]]
            solution[i] = candidate
        
        return Individual(self.tsp,order=solution)

    ### Mutation
    def mutation(self,indv):
        self.swap(indv)
    ### Mutation: swap
    def swap(self,indv):
        if random.random() < indv.alpha:
            i = random.randint(0,indv.size-1)
            j = random.randint(0,indv.size-1)
            indv.order[i], indv.order[j] = indv.order[j], indv.order[i]
    
    ### Elimination mu + lambda
    def elimination(self,offspring):
        combined = np.concatenate((self.population, offspring))
        combined = sorted(combined, key=lambda x: self.fitness(x))
        return combined[:self.p.lamb]
    
    # The evolutionary algorithm’s main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        distanceMatrix = None  	
        with open(filename) as file:
            distanceMatrix = np.loadtxt(file, delimiter=",")
        
        # Create TSP problem instance and set parameters
        self.tsp = TSPProblem(distanceMatrix)
        self.p = Parameters(lamb=100, mu=100, num_iters=100,k=10)
        self.iter = self.p.num_iters
        # Initialise population
        self.population = self.initPopulation()
        # add nearest neighbour to population
        self.population.append(self.nearestNeighbour())

        while(self.convergenceTest()):
            
            start = time.time()

            ### create offspring
            offspring = np.zeros(self.p.mu,dtype=Individual)
            for i in range(self.p.mu):
                p1 = self.selection()
                p2 = self.selection()
                offspring[i] = self.recombination(p1,p2)
                self.mutation(offspring[i])

            ### Mutate the original population
            for ind in self.population:
                self.mutation(ind)
            
            ### Elimination
            self.population = self.elimination(offspring)

            fitnesses  = np.array([self.fitness(individual) for individual in self.population])

            stop = time.time()

            meanObjective = np.mean(fitnesses)
            bestObjective = np.min(fitnesses)
            bestSolution = self.population[np.argmin(fitnesses)].order

            print("iteration: ",self.p.num_iters - self.iter, "    Mean Objective: ", 
                  meanObjective,"   Best Objective: ", bestObjective, "    Time: ", stop-start)

            timeLeft = self.reporter.report(meanObjective , bestObjective , bestSolution)
            if timeLeft < 0:
                break
    
        # Your code here.
        return 0

if __name__ == '__main__':
    algo = r0810938()
    algo.optimize("tour50.csv")



