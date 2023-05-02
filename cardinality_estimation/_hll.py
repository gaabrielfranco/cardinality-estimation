import hashlib
import math
from sys import getsizeof
import mmh3
import numpy as np
from ._constants import raw_estimate_data, bias_data, threshold_data
from ._base import BaseCardinalityEstimation

class HyperLogLog(BaseCardinalityEstimation):
    def __init__(self, p, hash="mmh3", random_state=42) -> None:
        if p < 4 or p > 16:
            raise ValueError("p must be between 4 and 16")
        self.p = p
        self.random_state = random_state
        self.m = 2**p
        self.buckets = np.zeros(self.m, dtype=int)
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

        self.hash_name = hash
        if self.hash_name == "mmh3":
            self.hash = lambda x: mmh3.hash(x, self.random_state, signed=False)
        elif self.hash_name == "sha256":
            self.hash = lambda x: int.from_bytes(hashlib.sha256(x).digest()[:4], byteorder="little")
        else:
            raise ValueError("Unknown hash function")

    def insert(self, x) -> None:
        # Convert to bytes
        if isinstance(x, int):
            x = x.to_bytes((x.bit_length() + 7) // 8, 'little')
        
        hash_x = self.hash(x)

        bucket = hash_x >> (32 - self.p)
        w = hash_x & ((1 << (32 - self.p)) - 1)

        #Count the number of leading zeros (considering 32 bits)
        leading_zeros = 32 - self.p - w.bit_length() + 1

        self.buckets[bucket] = max(self.buckets[bucket], leading_zeros)

    def get_estimate(self) -> int:
        E = self.alpha * (self.m ** 2 ) / sum(math.pow(2.0, -bucket) for bucket in self.buckets)
        if E <= 2.5 * self.m:
            V = np.sum(self.buckets == 0)
            if V != 0:
                # Linear counting
                return int(round(self.m * np.log(self.m / V)))
            else:
                return int(round(E))
        elif E <= 1/30 * 2**32:
            return int(round(E))
        else:
            return int(round(-2**32 * np.log(1 - (E / 2**32))))
        
    def get_size(self) -> int:
        return getsizeof(self.buckets)
    
    def merge(self, hll) -> None:
        if self.p != hll.p:
            raise ValueError("p must be the same")
        if self.self.hash_name != hll.self.hash_name:
            raise ValueError("hash function must be the same")
        
        self.buckets = np.maximum(self.buckets, hll.buckets)
               
class HyperLogLogPlusPlus(BaseCardinalityEstimation):
    def __init__(self, p, hash="mmh3",  random_state=42) -> None:
        if p < 4 or p > 18:
            raise ValueError("p must be between 4 and 18")
        self.p = p
        self.random_state = random_state
        self.m = 2**p
        self.buckets = np.zeros(self.m, dtype=int)
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)
        self.hash_name = hash

        if self.hash_name == "mmh3":
            self.hash = lambda x: mmh3.hash64(x, self.random_state, signed=False)[1]
        elif self.hash_name == "sha256":
            self.hash = lambda x: int.from_bytes(hashlib.sha256(x).digest()[:8], byteorder="little")
        else:
            raise ValueError("Unknown hash function")

    def linear_counting(self, m, V) -> int:
        return int(round(m * np.log(m / V)))
    
    def threshold(self, p) -> int:
        return threshold_data[p-4]
        
    def estimate_bias(self, E, p) -> int:
        for pos in range(len(raw_estimate_data[p-4])):
            if raw_estimate_data[p-4][pos] > E:
                break

        # Boundary cases (following C++ implementation)
        if pos == 0:
            return bias_data[p-4][pos]
        elif pos == len(raw_estimate_data[p-4]):
            return bias_data[p-4][pos-1]
        
        # Linear interpolation
        low_est = raw_estimate_data[p-4][pos-1]
        high_est = raw_estimate_data[p-4][pos]

        range_est = high_est - low_est
        scale = (E - low_est) / range_est

        low_bias = bias_data[p-4][pos-1]
        high_bias = bias_data[p-4][pos]

        bias_range = high_bias - low_bias
        est_bias = low_bias + (scale * bias_range)

        return est_bias
    
    def insert(self, x) -> None:
        # Convert to bytes
        if isinstance(x, int):
            x = x.to_bytes((x.bit_length() + 7) // 8, 'little')
        
        hash_x = self.hash(x)

        bucket = hash_x >> (64 - self.p)
        w = hash_x & ((1 << (64 - self.p)) - 1)

        #Count the number of leading zeros (considering 64 bits)
        leading_zeros = 64 - self.p - w.bit_length() + 1

        self.buckets[bucket] = max(self.buckets[bucket], leading_zeros)

    
    def get_estimate(self) -> int:
        E = self.alpha * (self.m ** 2 ) / sum(math.pow(2.0, -bucket) for bucket in self.buckets)

        # Estimate bias is already estimating the final value instead of returning the bias
        E_prime = E - self.estimate_bias(E, self.p) if E <= (5 * self.m) else E
        E_prime = int(round(E_prime))
        V = np.sum(self.buckets == 0)
        if V != 0:
            # Linear counting
            H = self.linear_counting(self.m, V)
        else:
            H = E_prime
        
        if H <= self.threshold(self.p):
            return H
        else:
            return E_prime
            
    def get_size(self) -> int:
        return getsizeof(self.buckets)

    def merge(self, hll) -> None:
        if self.p != hll.p:
            raise ValueError("p must be the same")
        if self.hash_name != hll.hash_name:
            raise ValueError("hash function must be the same")
        
        self.buckets = np.maximum(self.buckets, hll.buckets)