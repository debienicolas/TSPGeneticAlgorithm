from colorama import init
from matplotlib import axis
from r0123456 import *
from r0123456copy import *
import unittest
import numpy as np


class Test(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.p = Parameters(lamb=15, mu=15,num_iters=1500,k=9,alpha=0.05,alpha_sharing=0.20,sigmaPerc=1)
        self.algo = r0123456(self.p)
    
    # def standardiseTest(self):
    #     individual = np.array([1, 2,0, 3])
    #     indv_2 = np.array([0, 1, 2,3])
    #     self.assertEqual(self.algo.standardise(individual), np.array([0, 3,1,2]))
    #     self.assertEqual(self.algo.standardise(indv_2), np.array([0, 1, 2,3]))
    #     individ3 = np.array([2,1,3,0])
    #     self.assertEqual(self.algo.standardise(individ3), np.array([0,2,1,3]))
        
    def test_fitness_sharing(self):
        wandb.init(project="test")
        wandb.config.alpha_sharing = 2.5
        wandb.config.sigmaPerc = 1.0
        with open("Data/tour50.csv") as file:
            self.algo.distanceMatrix = np.loadtxt(file, delimiter=",")
            self.algo.length = self.algo.distanceMatrix.shape[0]
        
        population = np.array([initialize_legally(self.algo.distanceMatrix) for _ in range(10)])
        fitnesses = np.array([fitness(self.algo.distanceMatrix, individual) for individual in population])
        print(population.shape)
        print(fitnesses.shape)
        survivors = population[:2,:]
        print(survivors.shape)
        mod_fitnesses = self.algo.sharedFitnessWrapper(fitnesses, population, survivors,betaInit=1)
        print(mod_fitnesses.shape)
        print(mod_fitnesses)
        for i in range(10):
            print("Distance between survivor 1 and indiv: ", distance(survivors[0], population[i]))
            print("Distance between survivor 2 and indiv: ", distance(survivors[1], population[i]))
            print("Original fitness: ", fitnesses[i])
            print("Modified fitness: ", mod_fitnesses[i])

    
    def test_opt2swap(self):
        indv = np.array([0,1,2,3,4,5,6,7,8,9,10])
        new_indv = opt2swap(indv, 3, 7)
    
    def test_init_lists(self):
        with open("Data/tour50.csv") as file:
            self.algo.distanceMatrix = np.loadtxt(file, delimiter=",")
            self.algo.length = self.algo.distanceMatrix.shape[0]
        indv = initialize_legally(self.algo.distanceMatrix)
        pd = acc_fitness(self.algo.distanceMatrix, indv)
        rd = acc_fitness(self.algo.distanceMatrix, indv[::-1])
        self.assertEqual(pd[-1],fitness(self.algo.distanceMatrix, indv))
        self.assertEqual(rd[-1], fitness(self.algo.distanceMatrix, indv[::-1]))
        self.assertEqual(pd[1], self.algo.distanceMatrix[indv[0],indv[1]] )
        self.assertEqual(rd[1], self.algo.distanceMatrix[indv[-1],indv[-2]])  

        i = 10
        j = 20
        swapped = opt2swap(indv, i, j)
        origFit = fitness(self.algo.distanceMatrix, indv)
        swappedFit = fitness(self.algo.distanceMatrix, swapped)

        calculatedFitness = calc_two_opt_fit(self.algo.distanceMatrix, indv,pd,rd, i, j)


        

    
    def test_greedyInit(self):
        with open("Data/tour200.csv") as file:
            self.algo.distanceMatrix = np.loadtxt(file, delimiter=",")
            self.algo.length = self.algo.distanceMatrix.shape[0]
        indv = initialize_greedily(self.algo.distanceMatrix)
        

    
    def test_distance(self):
        def distance(indv1, indv2):
            edges1 = np.column_stack((indv1, np.roll(indv1, -1)))
            edges2 = np.column_stack((indv2, np.roll(indv2, -1)))
            similarity = np.count_nonzero((edges1[:,None] == edges2).all(-1).any(-1))
            return indv1.size - similarity
        
        indv1 = np.array([0, 1, 2,3])
        indv2 = np.array([2,3,1,0])
        self.assertEqual(distance(indv1, indv2), 3)
        indv2 = np.array([0, 1,2,3])
        self.assertEqual(distance(indv1, indv2), 0)
        indv2 = np.array([0,1,3,2])
        self.assertEqual(distance(indv1, indv2), 3)

    # def test_fitness(self):
    #     with open("Data/tour50.csv") as file:
    #         self.algo.distanceMatrix = np.loadtxt(file, delimiter=",")
    #         self.algo.length = self.algo.distanceMatrix.shape[0]
    #     for i in range(10000):
    #         individual = np.random.permutation(self.algo.distanceMatrix.shape[0])
    #         print(self.algo.fitness(individual))
    #         print(self.algo.fitness2(individual))
    #         self.assertEqual(self.algo.fitness(individual), self.algo.fitness2(individual))
        
    
    # def test_fitness(self):
    #     distanceMatrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    #     def fitness(individual):
    #         # calculate the fitness of the individual
    #         # the fitness is the total distance of the cycle
    #         indices = np.stack((individual,np.roll(individual,-1)),axis=1)
    #         return np.sum(distanceMatrix[indices[:,0],indices[:,1]])
        
    #     self.assertTrue(fitness([[0,1,2],[1,2,0]]).equal(np.array([6,6])))
    #     indv = [0, 1, 2]
    #     self.assertEqual(fitness(indv),6)
    #     indv = [1, 2, 0]
    #     self.assertEqual(fitness(indv),6)
        




if __name__ == "__main__":
    unittest.main()