import argparse
import os
from sys import getsizeof
import numpy as np
import pandas as pd
from cardinality_estimation import HyperLogLog, HyperLogLogPlusPlus, TextBookEstimation
import time

def ratio_metric(estimate, real):
    return estimate/real

# Argparse
parser = argparse.ArgumentParser(description="Synthetic experiments")
parser.add_argument("--max_range", "-mr", type=int, required=True)
parser.add_argument("--algorithm", "-a", type=str, required=True)
parser.add_argument("--p", "-p", type=int, required=False)
parser.add_argument("--hash", "-hs", type=str, required=False)
parser.add_argument("--eps", "-eps", type=float, required=False)
parser.add_argument("--delta", "-d", type=float, required=False)
args = parser.parse_args()

if args.algorithm == "tbe":
    alg_name = args.algorithm + "_" + str(args.eps) + "_" + str(args.delta)
else:
    alg_name = args.algorithm + "_" + str(args.p) + "_" + str(args.hash)

filename = f"experiments/synthetic/{args.max_range}_{alg_name}.csv"

if os.path.exists(filename):
    print(f"File {filename} already exists")
    exit()

df = pd.DataFrame(columns=["algorithm", "max_range", "real_cardinality", "estimated_cardinality", "ratio", "size", "insertion_time_mean", "insertion_time_std", "estimation_time"])

random = np.random.RandomState(args.max_range)

for i in range(10):
    D = list(range(args.max_range))
    # Adding duplicates
    D = D + random.choice(range(args.max_range), size=5000000 - args.max_range, replace=True).tolist()
    D = np.array(D)
    random.shuffle(D)

    real_cardinality = args.max_range

    SEED = random.randint(0, 1000000)

    # Estimators
    if args.algorithm == "hll":
        alg = HyperLogLog(p=args.p, hash=args.hash, random_state=SEED)
    elif args.algorithm == "hllpp":
        alg = HyperLogLogPlusPlus(p=args.p, hash=args.hash, random_state=SEED)
    elif args.algorithm == "tbe":
        alg = TextBookEstimation(eps=args.eps, delta=args.delta, m=len(D), random_state=SEED)

    print(f"Execution {i}:")
    print(args, "\n")

    insertion_time = []
    max_memory_usage = -1
    for x in D:
        start = time.time()
        alg.insert(x)
        # Insertion time in ms
        insertion_time.append((time.time() - start) * 1000)

        # Memory usage (we do that because of the TBE method. HLL/HLL++ memory usage is "constant", i.e., always O(m))
        max_memory_usage = max(max_memory_usage, alg.get_size())

    # Estimation
    start = time.time()
    estimated_cardinality = alg.get_estimate()
    estimation_time = (time.time() - start) * 1000

    # Size
    alg_size = max_memory_usage

    # Quality metric
    alg_ratio = ratio_metric(estimated_cardinality, real_cardinality)

    # Time
    alg_insertion_time_mean, alg_insertion_time_std = np.mean(insertion_time), np.std(insertion_time)

    df.loc[i] = [alg_name, args.max_range, real_cardinality, estimated_cardinality, alg_ratio, alg_size, alg_insertion_time_mean, alg_insertion_time_std, estimation_time]
df.to_csv(filename, index=False)


