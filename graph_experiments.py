import argparse
from copy import deepcopy
import re
import time

import numpy as np
import pandas as pd
from cardinality_estimation import HyperLogLogPlusPlus

def ratio_metric(estimate, real):
    return estimate/real

def merge_hll(hlls):
    new_hll = deepcopy(hlls[0])
    for hll in hlls[1:]:
        new_hll.merge(hll)
    return new_hll

# Argparse
parser = argparse.ArgumentParser(description="Synthetic experiments")
parser.add_argument("--algorithm", "-a", type=str, required=True)
parser.add_argument("--p", "-p", type=int, required=True)
parser.add_argument("--hash", "-hs", type=str, required=True)
parser.add_argument("--n_hlls", "-n", type=int, required=True)
parser.add_argument("--start", "-s", type=int, required=True)
parser.add_argument("--end", "-e", type=int, required=True)
parser.add_argument("--n_execution", "-ne", type=int, required=True)
args = parser.parse_args()

print(f"Algorithm: {args.algorithm}")
print(f"p: {args.p}")
print(f"Hash: {args.hash}")
print(f"n_hlls: {args.n_hlls}")
print(f"start: {args.start}")
print(f"end: {args.end}")
print(f"n_execution: {args.n_execution}")
print("\n------------------\n")

df = pd.DataFrame(columns=["n_execution", "algorithm", "p", "hash", "n_hlls", "real_cardinality", "estimated_cardinality", "ratio", "size", "insertion_time_mean", "insertion_time_std", "estimation_time"])
random = np.random.RandomState(args.n_execution)
random_seeds = random.randint(0, 1000000, args.n_hlls)

if args.algorithm == "hllpp":
    hllpps = [HyperLogLogPlusPlus(p=args.p, hash=args.hash, random_state=random_seeds[i]) for i in range(args.n_hlls)]
else:
    raise NotImplementedError

f = open("edges.txt", "r")
insertion_time = []
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

hllpp = merge_hll(hllpps)
real_cardinality = args.end - args.start

# Estimation
start_time = time.time()
estimated_cardinality = hllpp.get_estimate()
estimation_time = (time.time() - start_time) * 1000

# Size
alg_size = hllpp.get_size()

# Quality metric
alg_ratio = ratio_metric(estimated_cardinality, real_cardinality)

# Time
alg_insertion_time_mean, alg_insertion_time_std = np.mean(insertion_time), np.std(insertion_time)

alg_name = args.algorithm + "_" + str(args.p) + "_" + str(args.hash) + "_" + str(args.n_hlls)

filename = f"experiments/graph/{alg_name}_{args.start}_{args.end}_{args.n_execution}.csv"

df.loc[0] = [args.n_execution, args.algorithm, args.p, args.hash, args.n_hlls, real_cardinality, estimated_cardinality, alg_ratio, alg_size, alg_insertion_time_mean, alg_insertion_time_std, estimation_time]
df.to_csv(filename, index=False)
