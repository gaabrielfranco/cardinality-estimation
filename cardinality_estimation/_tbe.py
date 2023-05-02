import math
from sys import getsizeof
import numpy as np
from ._base import BaseCardinalityEstimation
from memory_profiler import profile

# Distinct Elements in Streams: An Algorithm for the (Text) Book implementation
class TextBookEstimation(BaseCardinalityEstimation):
    def __init__(self, eps, delta, m, random_state=None) -> None:
        self.eps = eps
        self.delta = delta
        self.m = m
        self.X = set()
        self.thresh = (12/(eps**2)) * math.log((8*self.m) / delta)
        self.p = 1
        self.random = np.random.RandomState(random_state)

    def insert(self, x) -> None:
        self.X.discard(x)
        if self.random.random() < self.p:
            self.X.add(x)
        if len(self.X) >= self.thresh:
            # Throw away each element with probability 1/2
            discard_elems = set()
            for x in self.X:
                if self.random.random() < 0.5:
                    discard_elems.add(x)
            self.X -= discard_elems
            self.p /= 2

            if len(self.X) >= self.thresh:
                raise Exception("Error: X.size() > thresh")

    def get_estimate(self) -> int:
        return int(round(len(self.X) / self.p))
    
    def get_size(self) -> int:
        return getsizeof(self.X)