from r0123456 import *
import unittest
import numpy as np


class Test(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.p = Parameters(10, 10, 10, 10, 0.5)
        self.algo = r0123456(self.p)
    
    def test_distance(self):
        distanceMatrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        tsp = TSPProblem(distanceMatrix)
        indv1 = Individual(tsp, alpha=0.5, order=np.array([0, 1, 2]))
        indv2 = Individual(tsp, alpha=0.5, order=np.array([1, 2, 0]))
        self.assertEqual(self.algo.distance(indv1, indv2),0)
        indv2 = Individual(tsp, alpha=0.5, order=np.array([2, 1, 0]))
        self.assertEqual(self.algo.distance(indv1, indv2),3)


if __name__ == "__main__":
    unittest.main()