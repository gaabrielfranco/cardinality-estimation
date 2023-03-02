import random
import intset
import math

# Distinct Elements in Streams: An Algorithm for the (Text) Book implementation

random.seed(326178)

# Params
eps, delta = 0.1, 1e-5
m = 1000000

# Input (stream of integers)
D = random.sample(range(m), k=m)

p = 1
X = intset.IntSet()
thresh = 12/(eps**2) * math.log(8*m / delta)

for i in range(m):
    X = X.discard(D[i])
    if random.random() <= p:
        X = X.insert(D[i])
    if X.size() > thresh:
        # Throw away each element with probability 1/2
        for x in X:
            if random.random() <= 0.5:
                X = X.discard(x)
        p /= 2

        if X.size() > thresh:
            raise Exception("Error: X.size() > thresh")

print("Estimation: %d" % (X.size() / p))
print("Real: %d" % len(set(D)))
