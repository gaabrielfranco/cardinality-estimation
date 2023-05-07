import argparse
from copy import deepcopy
import os
import re
import time

import numpy as np
import pandas as pd
from cardinality_estimation import HyperLogLogPlusPlus, TextBookEstimation

def ratio_metric(estimate, real):
    return estimate/real

def merge_hll(hlls):
    new_hll = deepcopy(hlls[0])
    for hll in hlls[1:]:
        new_hll.merge(hll)
    return new_hll

# Argparse
parser = argparse.ArgumentParser(description="Synthetic experiments")
parser.add_argument("--algorithm", "-a", type=str, choices=["hllpp", "tbe"], required=True)
parser.add_argument("--p", "-p", type=int, required=False)
parser.add_argument("--hash", "-hs", type=str, required=False)
parser.add_argument("--n_hlls", "-n", type=int, required=False)
parser.add_argument("--eps", "-eps", type=float, required=False)
parser.add_argument("--delta", "-d", type=float, required=False)
parser.add_argument("--start", "-s", type=int, required=True)
parser.add_argument("--end", "-e", type=int, required=True)
parser.add_argument("--n_execution", "-ne", type=int, required=True)
args = parser.parse_args()

if args.algorithm == "hllpp":
    alg_name = args.algorithm + "_" + str(args.p) + "_" + str(args.hash) + "_" + str(args.n_hlls)
else:
    alg_name = args.algorithm + "_" + str(args.eps) + "_" + str(args.delta)

filename = f"experiments/graph/{alg_name}_{args.start}_{args.end}_{args.n_execution}.csv"

if os.path.exists(filename):
    print(f"File {filename} already exists\n")
    exit()

if args.algorithm == "hllpp":
    print(f"Algorithm: {args.algorithm}")
    print(f"p: {args.p}")
    print(f"Hash: {args.hash}")
    print(f"n_hlls: {args.n_hlls}")
    print(f"start: {args.start}")
    print(f"end: {args.end}")
    print(f"n_execution: {args.n_execution}")
    print("\n------------------\n")
else:
    print(f"Algorithm: {args.algorithm}")
    print(f"eps: {args.eps}")
    print(f"delta: {args.delta}")
    print(f"start: {args.start}")
    print(f"end: {args.end}")
    print(f"n_execution: {args.n_execution}")
    print("\n------------------\n")

df = pd.DataFrame(columns=["n_execution", "algorithm", "p", "hash", "n_hlls", "real_cardinality", "estimated_cardinality", "ratio", "size", "insertion_time_mean", "insertion_time_std", "estimation_time", "eps", "delta"])
random = np.random.RandomState(args.n_execution)

if args.algorithm == "hllpp":
    random_seeds = random.randint(0, 1000000, args.n_hlls)
    hllpps = [HyperLogLogPlusPlus(p=args.p, hash=args.hash, random_state=random_seeds[i]) for i in range(args.n_hlls)]
else:
    estimator = TextBookEstimation(eps=args.eps, delta=args.delta, m=args.end - args.start, random_state=random.randint(0, 1000000))

f = open("edges.txt", "r")
insertion_time = []
max_algorithm_size = -1

if args.algorithm == "hllpp":
    for idx, line in enumerate(f):
        if idx > args.start:
            line = re.sub(r"\n", "", line)
            line = re.sub(r"\t", ",", line)
            if args.hash == "sha256":
                line = line.encode("utf-8")
            start_time = time.time()
            hllpps[idx%args.n_hlls].insert(line)
            # Sample insertion time in ms
            if idx % 1000 == 0:
                insertion_time.append((time.time() - start_time) * 1000)

        if idx == args.end:
            break

    estimator = merge_hll(hllpps)
else:
    for idx, line in enumerate(f):
        if idx > args.start:
            line = re.sub(r"\n", "", line)
            line = re.sub(r"\t", ",", line)
            start_time = time.time()
            estimator.insert(line)
            # Sample insertion time in ms
            if idx % 1000 == 0:
                insertion_time.append((time.time() - start_time) * 1000)
            
            if estimator.get_size() > max_algorithm_size:
                max_algorithm_size = estimator.get_size()

        if idx == args.end:
            break

real_cardinality = args.end - args.start

# Estimation
start_time = time.time()
estimated_cardinality = estimator.get_estimate()
estimation_time = (time.time() - start_time) * 1000

# Size
alg_size = max_algorithm_size if args.algorithm == "tbe" else estimator.get_size()

# Quality metric
alg_ratio = ratio_metric(estimated_cardinality, real_cardinality)

# Time
alg_insertion_time_mean, alg_insertion_time_std = np.mean(insertion_time), np.std(insertion_time)

if args.algorithm == "hllpp":
    df.loc[0] = [args.n_execution, args.algorithm, args.p, args.hash, args.n_hlls, real_cardinality, estimated_cardinality, alg_ratio, alg_size, alg_insertion_time_mean, alg_insertion_time_std, estimation_time, None, None]
else:
    df.loc[0] = [args.n_execution, args.algorithm, None, None, None, real_cardinality, estimated_cardinality, alg_ratio, alg_size, alg_insertion_time_mean, alg_insertion_time_std, estimation_time, args.eps, args.delta]

df.to_csv(filename, index=False)
