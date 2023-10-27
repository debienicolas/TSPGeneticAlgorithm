import numpy as np
import random

# load the distance matrix
file = open("tour50.csv")
distanceMatrix = np.loadtxt(file, delimiter=",") 


class Individual:
    def __init__(self,tsp,order=None,alpha=None):

        self.alpha = max(0.1,0.1+0.2*random.random()) 
        self.order = np.random.permutation(range(tsp.nCities))
        self.size = len(self.order)

        if order is not None:
            self.order = order
            self.size = len(self.order)
        if alpha is not None:
            self.alpha = alpha

class TSPProblem:
    def __init__(self,distanceMatrix):
        self.distanceMatrix = distanceMatrix
        self.nCities = len(distanceMatrix)

class Parameters:
    def __init__(self,lamb, mu, num_iters, k):
        self.lamb = lamb
        self.mu = mu
        self.num_iters = num_iters
        self.k = k

def evolutionaryAlgorithm(tsp,p):

    population = initializePopulation(tsp,p)

    for i in range(p.num_iters):
        offspring = [0] * p.mu
        for j in range(p.mu):
            p1 = selection(tsp,population,p.k)
            p2 = selection(tsp,population,p.k)
            offspring[j] = recombination(tsp,p1,p2)
            mutate(offspring[j])

        for ind in population:
            mutate(ind)
        
        # elemination
        population = elimination(population,offspring,p.lamb)

        fitnesses = [fitness(ind) for ind in population]
        print("mean fitness: ", np.mean(fitnesses), " best fitness: ", np.min(fitnesses))


def fitness(tsp,individual:Individual):
    # calculate the fitness of the individual
    # the fitness is the total distance of the cycle
    distance = 0
    for i in range(individual.size - 1):
        distance += tsp.distanceMatrix[individual.order[i]][individual.order[i+1]]
    # remember to add the distance from the last city to the first city
    distance += distanceMatrix[individual.order[individual.size - 1]][individual.order[0]]
    return distance

def initializePopulation(tsp,p):
    return [Individual(tsp) for i in range(p.lamb)]

def selection(population,k):
    candidates = random.sample(population,k)
    return candidates[np.argmin([fitness(ind) for ind in candidates])]

def mutate(individual:Individual):
    # mutate the individual by swapping two cities
    # with probability alpha
    if random.random() < individual.alpha:
        i = random.randint(0,individual.size-1)
        j = random.randint(0,individual.size-1)
        individual.order[i], individual.order[j] = individual.order[j], individual.order[i]

# PMX recombination
def recombination(tsp,p1,p2):
    # select two random indices
    i = random.randint(0,tsp.nCities-1)
    j = random.randint(0,tsp.nCities-1)
    if i > j:
        i,j = j,i
    print("i: ", i, " j: ", j)
    # create a child with the cities between i and j from parent 1
    childOrder = [None] * tsp.nCities
    childOrder[i:j] = p1.order[i:j]
    print("childOrder: ", childOrder)

    for city in p2.order[i:j]:
        if city not in childOrder:
            childOrder[np.where(p1.order == city)[0]] = city
    print("childOrder: ", childOrder)

    # copy remaining cities over 
    for (elem, i) in enumerate(childOrder):
        if elem is None:
            childOrder[i] = p2.order[i]
        
    return Individual(tsp,childOrder)
            



# generate random distance matrix with size 10 with 0 ono the diagonal
distanceMatrix = np.random.randint(0,10,(5,5))
np.fill_diagonal(distanceMatrix,0)
print(distanceMatrix)



tsp = TSPProblem(distanceMatrix)
p1 = Individual(tsp)
p2 = Individual(tsp)
print("Parent 1: ", p1.order)
print("Parent 2: ", p2.order)
child = recombination(tsp,p1,p2)
print("Child: ", child.order)