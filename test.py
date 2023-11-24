from r0123456 import *
import unittest
import numpy as np


class Test(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.p = Parameters(10, 10, 10, 10, 0.5)
        #self.algo = r0123456(self.p)
    
    # def test_distance(self):
    #     distanceMatrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    #     tsp = TSPProblem(distanceMatrix)
    #     indv1 = Individual(tsp, alpha=0.5, order=np.array([0, 1, 2]))
    #     indv2 = Individual(tsp, alpha=0.5, order=np.array([1, 2, 0]))
    #     self.assertEqual(self.algo.distance(indv1, indv2),0)
    #     indv2 = Individual(tsp, alpha=0.5, order=np.array([2, 1, 0]))
    #     self.assertEqual(self.algo.distance(indv1, indv2),3)
    
    def test_fitness(self):
        def fitness(self,distanceMatrix,individual):
            # calculate the fitness of the individual
            # the fitness is the total distance of the cycle
            indices = np.stack((individual,np.roll(individual,-1)),axis=1)
            return np.sum(distanceMatrix[indices[:,0],indices[:,1]])
        distanceMatrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        indv = [0, 1, 2]
        self.assertEqual(fitness(self,distanceMatrix,indv),6)
        indv = [1, 2, 0]
        self.assertEqual(fitness(self,distanceMatrix,indv),6)
        




if __name__ == "__main__":
    unittest.main()