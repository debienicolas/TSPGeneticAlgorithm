from cv2 import distanceTransform
import Reporter
import numpy as np
import random
from timeit import default_timer as timer
from numba import jit, njit

import math

from numba import jit, prange

INF = 1000000000000000


@njit
def standardise(indv):
    index = np.where(indv == 0)[0][0]
    return np.concatenate((indv[index:],indv[:index]))


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
        else:
            distance += distanceMatrix[i[0],i[1]]
    return distance

@njit
def distance(ind1,ind2):
    size = ind1.size
    succesors = np.empty(size, dtype=np.int32)
    for i in prange(size):
        succesors[ind2[i]] = ind2[(i+1)%size]
    count_nonshared_edges = 0
    for i in prange(size):
        j = (i+1)%size
        if ind1[j] != succesors[ind1[i]]:
            count_nonshared_edges += 1
    return count_nonshared_edges


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

@njit
def opt2swap(indv1, i, j):
    new_order = np.empty_like(indv1)
    # take route[0] to route[i] and add them in order to new_order
    new_order[:i] = indv1[:i]
    # take route[i] to route[j] and add them in reverse order to new_order
    new_order[i:j+1] = indv1[j:i-1:-1]
    # take route[j+1] to end and add them in order to new_order
    new_order[j+1:] = indv1[j+1:]
    return new_order

@njit
def opt2swapSingle(indv1,i,j):
    # swap the two cities
    indv1[i], indv1[j] = indv1[j], indv1[i]
    return indv1
    
    

@njit 
def singleFitness(distanceMatric,i,j):
    distance = distanceMatric[i,j]
    if distance == np.inf:
        return INF
    return distance
@njit
def acc_fitness(distM,indv):
    size = len(indv)
    acc = np.empty(size+1,dtype=float)
    acc[0] = 0
    for i in range(size-1):
        a1 = indv[i]
        a2 = indv[i+1]
        dist = singleFitness(distM,a1,a2)
        acc[i+1] = acc[i] + dist
    acc[size] = acc[size-1] + singleFitness(distM,indv[size-1],indv[0])
    return acc

@njit
def calc_two_opt_fit(dm ,route, pd: np.ndarray, rd: np.ndarray, i:int, j: int) -> float:
    orig = pd[-1]
    #assert pd[-1] == fitness(dm,route)
    
    # deleted arcs distances
    t1 = dm[route[i-1]][route[i]]
    t2 = dm[route[j]][route[j+1]]
    # added arcs distances
    t3 = dm[route[i-1]][route[j]]
    t4 = dm[route[i]][route[j+1]]

    after = orig - t1 - t2 + t3 + t4
    pd_del = pd[j] - pd[i]
    n = len(route)
    rd_add = rd[n - 1 - i] - rd[n - 1 - j]
    after = after - pd_del + rd_add
    return after


# def randomOpt2(distM,indv, percentage=0.9):
#     best_fitness = fitness(distM,indv)
#     best_indv = indv
#     size = indv.size
#     # subsample from 1 to size-2
#     subsample = np.random.choice(size-3,int(percentage*size),replace=False)
#     subsample += 1
#     pd = acc_fitness(distM,indv)
#     rd = acc_fitness(distM,indv[::-1])
#     for i in range(1,size-2):
#         for j in range(i+1,size-1):
#             if np.random.random() < percentage:
#                 new_fitness = calc_two_opt_fit(distM,indv,pd,rd,i,j)
#                 if new_fitness < best_fitness and not np.isinf(new_fitness) and new_fitness < INF:
#                     best_indv = opt2swap(indv,i,j)
#                     best_fitness = fitness(distM,best_indv)

#     return best_indv

@njit
def opt2Sample(distM,indv,percent= 0.3):
    best_indv = indv
    current_best = fitness(distM,indv)
    size = indv.size
    subsample_1 = np.random.choice(size-3,int(percent*size),replace=False)
    subsample_1 += 1
    for i in subsample_1:
        for j in range(i+1,size-2):
            lengthDelta = -distM[indv[i-1],indv[i]] - distM[indv[j],indv[(j+1)]] + distM[indv[i],indv[j+1]] + distM[indv[i-1],indv[j]]
            if lengthDelta < 0:
                new_order = opt2swap(indv,i,j)
                new_fitness = fitness(distM,new_order)
                if new_fitness < current_best:
                    best_indv = new_order
                    current_best = new_fitness
    return best_indv


@njit
def opt2(distM, indv):
    size = indv.size
    best_indiv = indv
    current_best = fitness(distM, indv)
    for i in prange(1,size - 2):
        for j in prange(i + 1, size-1):
            lenghtDelta = -distM[indv[i-1],indv[i]] - distM[indv[j],indv[(j+1)]] + distM[indv[i],indv[j+1]] + distM[indv[i-1],indv[j]]
            if lenghtDelta < 0:
                new_order = opt2swap(indv,i,j)
                new_fitness = fitness(distM,new_order)
                if new_fitness < current_best and not np.isinf(new_fitness) and new_fitness < INF:
                    best_indiv = new_order
                    current_best = new_fitness
    return best_indiv

@njit
def twoOpt(distM,indv):
    size = indv.size
    best_indiv = indv
    current_best = fitness(distM, indv)
    for i in prange(1,size - 2):
        for j in prange(i + 1, size-1):
            lenghtDelta = -distM[indv[i-1],indv[i]]- distM[indv[i],indv[i+1]] - distM[indv[j],indv[(j+1)]]-distM[indv[j],indv[j+1]] + distM[indv[i],indv[j+1]] + distM[indv[j-1],indv[i]] + distM[indv[i-1],indv[j]] + distM[indv[j],indv[i+1]]
            if lenghtDelta < 0 and not np.isinf(lenghtDelta):
                new_order = opt2swapSingle(indv,i,j)
                new_fitness = fitness(distM,new_order)
                if new_fitness < current_best and not np.isinf(new_fitness) and new_fitness < INF:
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
@njit
def select_next_city(current_city, parent1, parent2, visited ,distance_matrix):
    idx_p1 = np.where(parent1 == current_city)[0][0]
    idx_p2 = np.where(parent2 == current_city)[0][0]

    next_city_p1 = parent1[(idx_p1 + 1) % len(parent1)]
    next_city_p2 = parent2[(idx_p2 + 1) % len(parent2)]

    if visited[next_city_p1] and visited[next_city_p2]:
        # Both are visited, choose the a random unvisited city
        unvisited = np.where(visited == False)[0]
        return np.random.choice(unvisited)
    elif visited[next_city_p1]:
        return next_city_p2
    elif visited[next_city_p2]:
        return next_city_p1
    else:
        # Neither are visited, choose the closest
        if distance_matrix[current_city, next_city_p1] < distance_matrix[current_city, next_city_p2]:
            return next_city_p1
        else:
            return next_city_p2

@njit
def scx(parent1, parent2,offspring,visited_cities,distance_matrix):
    current_city = np.random.choice(parent1)  

    offspring[0] = current_city
    visited_cities[current_city] = True

    for i in range(1,len(parent1)):
        next_city = select_next_city(current_city, parent1, parent2,visited_cities,distance_matrix)
        visited_cities[next_city] = True
        offspring[i] = next_city

    return offspring


class r0810938:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.iter = 0
        self.population = None
        self.timings = {}
        self.used_dist_map = 0
    

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

    
    # Mutation wrapper function
    def mutation(self, individual, alpha):
        before_mut = individual.copy()
        if np.random.random() < alpha:
            inversionMutation(individual)
            assert np.equal(before_mut,individual).all() == False
        
    

    def recombination(self,parent1,parent2):
        return basicrossover(parent1,parent2,np.zeros(parent1.size,dtype=int))
        
    
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
    
    def sharedFitnessWrapper(self,orig_fitnesses, X, pop, betaInit = 1):
        modFitnesses = np.zeros(X.shape[0])
        max_distance = self.length
        
        alpha = self.p.alpha_sharing
        sigma = self.p.sigmaPerc * max_distance

        for i,x in enumerate(X):
            if orig_fitnesses[i] == np.inf:
                modFitnesses[i] = np.inf
                continue
            distances = np.zeros(pop.shape[0])
            for j,y in enumerate(pop):
                    distances[j] = distance(x,y)
            onePlusBeta = betaInit
            within_sigma = distances <= sigma
            onePlusBeta = betaInit + np.sum(1 - (distances[within_sigma] / sigma) ** alpha)
            # add to fitness hashmap
            fitnessval = orig_fitnesses[i]
            modFitnesses[i] = fitnessval *onePlusBeta** np.sign(fitnessval)
        
        return modFitnesses

    def sharedElimination(self, population, offspring):
        combined = np.concatenate((population,offspring), axis=0)
        survivors = np.zeros((self.p.lamb,self.length),dtype=int)
        fitnesses = np.array([fitness(self.distanceMatrix,ind) for ind in combined])
        for i in range(self.p.lamb):
            if i == 0:
                best_idx = np.argmin(fitnesses)
                survivors[i] = combined[best_idx]
                np.delete(combined,best_idx,0)
            else :
                # instead of calculating the updated fitness for all, only do it for the top 50% of the population ? 
                # take random subset of the combined population
                fpen = self.sharedFitnessWrapper(fitnesses,combined, survivors[0:i,:],betaInit=1)
                idx = np.argmin(fpen)
                #idx = current_best_idx
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
    
    def sharedEliminDecay(self,timeLeft):
        timePassed = 300 - timeLeft
        return math.exp(-self.p.sharedElimDecay_alpha * (((timePassed)/300)**self.p.sharedElimDecay_n)) - self.p.sharedElimDecay_c
    
    def LSOGrowth(self,timeLeft):
        timePassed = 300 - timeLeft
        return 1 - math.exp(-self.p.LSO_alpha * ((timePassed/300)**self.p.LSO_n)) + self.p.LSO_c

    def lambDecay(self,timeLeft):
        timePassed = 300 - timeLeft
        lamb_alpha = -math.log(self.p.lambEnd/self.p.lambStart)
        return int(self.p.lambStart*math.exp(-lamb_alpha * ((timePassed/300)**self.p.lambDecay_n)))
    
    def mutationDecay(self,timeLeft):
        timePassed = 300 - timeLeft
        alpha = -math.log(self.p.alphaEnd/self.p.alphaStart)
        return self.p.alphaStart*math.exp(-alpha * ((timePassed/300)**self.p.alphaDecay_n))

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
       
        self.p = Parameters(self.length)

        ### Population initialization ###
        start = timer()
        population = initializePopulation(distanceMatrix, self.p.lamb,self.p.percent_greedy_init)
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
        
        bestSolution = np.empty((self.length),dtype=int)
    
        while(self.convergenceTest()):
            # update parameter control
            
            self.lsogrowth = self.LSOGrowth(self.timeLeft)
            self.elimdecay = self.sharedEliminDecay(self.timeLeft)
            self.p.lamb = self.lambDecay(self.timeLeft)
            self.p.mu = self.p.lamb
            self.p.alpha = self.mutationDecay(self.timeLeft)
            self.p.lamb = self.lambDecay(self.timeLeft)
            self.p.mu = self.p.lamb
            ### Create offspring population ### 
            start = timer()
            offspring = np.zeros((self.p.mu,self.length), dtype=int)
            alphas_offspring = np.zeros((self.p.mu),dtype=float)
            for i in range(self.p.mu):
                alphas_offspring[i] = max(self.p.alpha, self.p.alpha + 0.1 * np.random.random())

            for i in range(self.p.mu):
                p1 = self.selection(population)
                p2 = self.selection(population)
                
                offspring[i] = self.recombination(p1,p2)
                
                self.mutation(offspring[i],alphas_offspring[i])
                ### apply local search operator to the offspring ###
                if self.lsogrowth > np.random.random():
                    offspring[i] = twoOpt(self.distanceMatrix,offspring[i])
            end = timer()
            self.timings["Create offspring + LSO"] = end - start


            ### Mutate the original population and apply local search operator ###
            start = timer()

            for i,ind in enumerate(population):
                if not np.equal(ind,bestSolution).all():
                    self.mutation(ind,alphas[i])
                if self.lsogrowth > np.random.random():
                    population[i] = twoOpt(self.distanceMatrix,ind)
                            
            end = timer()
            self.timings["Mutation + LSO"] = end - start

            ### Shared fitness elimination ###
            start = timer()
            if self.elimdecay > np.random.random():
                population = self.sharedElimination(population,offspring)          
            else:
                population = self.elimination(population,offspring)
            self.timings["Elimination"] = timer() - start

            # calculate fintesses
            fitnesses = np.zeros((self.p.lamb))
            for i in range(self.p.lamb):
                fitnesses[i] = fitness(self.distanceMatrix,population[i])
            
            meanObjective = np.mean(fitnesses)
            bestObjective = np.min(fitnesses)
            self.fitnesses.append(bestObjective)
            bestSolution = population[np.argmin(fitnesses)]
            diversityScore = len(set(fitnesses))

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            self.timeLeft = timeLeft
            print("Mean objective: ", round(meanObjective,2),"     Best Objective: ", round(bestObjective,2),"     Iteration: ",self.iter, "     Diversity score: ", diversityScore, "     Time left: ", round(timeLeft,2))
            
            if timeLeft < 0:
                break


        return bestObjective



class Parameters:
    def __init__(self, toursize):
        self.tour = toursize
        
        self.k = 5
        self.alpha_sharing = 0.25
        self.sigmaPerc = 0.5
        self.sharedElimDecay_alpha = 1
        self.sharedElimDecay_n = 15
        self.sharedElimDecay_c = 0

        self.LSO_alpha = 5
        self.LSO_n = 4
        self.LSO_c = 0.2
        self.LSOPercent = 0.1

        self.alpha = 0.1
        self.alphaStart = 0.3
        self.alphaEnd = 0.1
        self.alphaDecay_n = 0.4
        
        self.percent_greedy_init = 0.2
        self.lambStart = 60#80
        self.lambEnd = 20 #40
        self.lambDecay_n = 0.8
        self.lamb = self.lambStart
        self.mu = self.lamb

        




if __name__ == "__main__":


    assert True == True

    

    
    

