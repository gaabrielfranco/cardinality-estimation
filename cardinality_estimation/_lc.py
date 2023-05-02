import hashlib
import numpy as np
from ._base import BaseCardinalityEstimation
import mmh3

class LinearCounting(BaseCardinalityEstimation):
    def __init__(self, k, hash="mmh3", random_state=42) -> None:
        self.buckets = np.zeros(k, dtype="int8")
        self.k = k
        self.random_state = random_state
        if hash == "mmh3":
            self.hash = lambda x: mmh3.hash64(x, self.random_state, signed=False)[1]
        elif hash == "sha256":
            self.hash = lambda x: int.from_bytes(hashlib.sha256(x).digest()[:8], byteorder="little")
        else:
            raise ValueError("Unknown hash function")
    
    def insert(self, x) -> None:
        # Convert to bytes
        if isinstance(x, int):
            x = x.to_bytes((x.bit_length() + 7) // 8, 'little')
        
        hash_x = self.hash(x) % self.k
        self.buckets[hash_x] = 1

    def get_estimate(self) -> int:
        empty_buckets = np.sum(self.buckets == 0)
        if empty_buckets == 0:
            return -1# TODO: What to do here?
        else:
            return int(round(self.k * np.log(self.k / empty_buckets)))