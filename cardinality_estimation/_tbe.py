import math
from sys import getsizeof
import numpy as np
from ._base import BaseCardinalityEstimation

# Distinct Elements in Streams: An Algorithm for the (Text) Book implementation
class TextBookEstimation(BaseCardinalityEstimation):
    """
    TextBookEstimation algorithm for cardinality estimation.
    """
    def __init__(self, eps, delta, m, random_state=None) -> None:
        """
        Parameters
        ----------
        eps : float
            Error bound.
        delta : float
            Failure probability.
        m : int
            Size of the stream.
        random_state : int, optional
            Random seed.
        """
        
        self.eps = eps
        self.delta = delta
        self.m = m
        self.X = set()
        self.thresh = (12/(eps**2)) * math.log((8*self.m) / delta)
        self.p = 1
        self.random = np.random.RandomState(random_state)

    def insert(self, x) -> None:
        """
        Insert an element into the TextBookEstimation.

        Parameters
        ----------
        x : int
            Element to insert.

        Raises
        ------
        Exception
            If X.size() > thresh (Failure Event).
        """

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
                raise Exception("Error: X.size() > thresh - Failure Event")

    def get_estimate(self) -> int:
        """
        Get the cardinality estimate.

        Returns
        -------
        int
            Cardinality estimate.
        """
        return int(round(len(self.X) / self.p))
    
    def get_size(self) -> int:
        """
        Get the size of the TextBookEstimation in bytes.

        Returns
        -------
        int
            Size of the TextBookEstimation in bytes.
        """
        return getsizeof(self.X)