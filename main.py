from sys import getsizeof
import numpy as np
from cardinality_estimation import HyperLogLog, HyperLogLogPlusPlus, LinearCounting, TextBookEstimation
import time
from memory_profiler import profile
import mmh3

def ratio_metric(estimate, real):
    return estimate/real

random = np.random.RandomState(326178)

D = random.choice(range(80000), size=1000000, replace=True)

hll = HyperLogLog(p=16)
hllpp = HyperLogLogPlusPlus(p=16, hash="mmh3")
tbe = TextBookEstimation(eps=0.1, delta=0.1, m=1000000, random_state=326178)

hllpp_insertion_time = []
tbe_insertion_time = []
for x in D:
    hll.insert(x)
    start = time.time()
    hllpp.insert(x)
    hllpp_insertion_time.append(time.time() - start)
    start = time.time()
    tbe.insert(x)
    tbe_insertion_time.append(time.time() - start)


real = len(set(D))

start = time.time()
hllpp_estimate = hllpp.get_estimate()
print("HLL++ estimation time: %f (ms)" % ((time.time() - start) * 1000))
print()

print("Estimation (HLL): %d" % hll.get_estimate())
print("Estimation (HLL++): %d" % hllpp_estimate)
# print("Estimation (LC): %d" % lc.get_estimate())
print("Estimation (TBE): %d" % tbe.get_estimate())
print("Real: %d" % real)
print("Estimation (HLL++) ratio: %f" % ratio_metric(hllpp_estimate, real))

print("\nHLL++ insertion time: %f +- %f" % (np.mean(hllpp_insertion_time), np.std(hllpp_insertion_time)))
print("TBE insertion time: %f +- %f" % (np.mean(tbe_insertion_time), np.std(tbe_insertion_time)))

print("HLL++ size: %d" % hllpp.get_size())
print("TBE size: %d" % tbe.get_size())


# print()
# print(h.heap())
