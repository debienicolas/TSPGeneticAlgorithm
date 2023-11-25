from matplotlib import axis
from r0123456 import *
import unittest
import numpy as np


class Test(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.p = Parameters(10, 10, 10, 10, 0.5)
        self.algo = r0123456(self.p)
    
    # def standardiseTest(self):
    #     individual = np.array([1, 2,0, 3])
    #     indv_2 = np.array([0, 1, 2,3])
    #     self.assertEqual(self.algo.standardise(individual), np.array([0, 3,1,2]))
    #     self.assertEqual(self.algo.standardise(indv_2), np.array([0, 1, 2,3]))
    #     individ3 = np.array([2,1,3,0])
    #     self.assertEqual(self.algo.standardise(individ3), np.array([0,2,1,3]))
    
    def test_distance(self):
        def distance(indv1, indv2):
            print(indv1, indv2)
            edges1 = np.column_stack((indv1, np.roll(indv1, -1)))
            edges2 = np.column_stack((indv2, np.roll(indv2, -1)))
            similarity = np.count_nonzero((edges1[:,None] == edges2).all(-1).any(-1))
            return indv1.size - similarity
        
        indv1 = np.array([0, 1, 2,3])
        indv2 = np.array([2,3,1,0])
        self.assertEqual(distance(indv1, indv2), 3)
        indv2 = np.array([0, 1, 2,3])
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