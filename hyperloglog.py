import math
import mmh3
import numpy as np

class LinearCounting():
    def __init__(self, k) -> None:
        self.buckets = np.zeros(k, dtype="int8")
        self.k = k
    
    def insert(self, x) -> None:
        # Convert to bytes
        if isinstance(x, int):
            x = x.to_bytes((x.bit_length() + 7) // 8, 'little')
        
        hash_x = mmh3.hash(x, signed=False) % self.k
        self.buckets[hash_x] = 1

    def get_estimate(self) -> int:
        empty_buckets = np.sum(self.buckets == 0)
        if empty_buckets == 0:
            return # TODO: What to do here?
        else:
            return int(round(self.k * np.log(self.k / empty_buckets)))
    
class HyperLogLog():
    def __init__(self, p) -> None:
        self.p = p
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

    def insert(self, x) -> None:
        # Convert to bytes
        if isinstance(x, int):
            x = x.to_bytes((x.bit_length() + 7) // 8, 'little')
        
        hash_x = mmh3.hash(x, signed=False)

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
            return int(round(-2**32 * np.log(1 - E / 2**32)))


# Testing HyperLogLog
random = np.random.RandomState(326178)

# Getting random data
D = random.randint(0, 10000, size=1000000, dtype=int)
D = [x.item() for x in D]

hll = HyperLogLog(p=16)
lc = LinearCounting(k=10000)

for x in D:
    hll.insert(x)
    lc.insert(x)

print("Estimation (HLL): %d" % hll.get_estimate())
print("Estimation (LC): %d" % lc.get_estimate())
print("Real: %d" % len(set(D)))
