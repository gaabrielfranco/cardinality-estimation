from copy import deepcopy
import numpy as np
import pandas as pd
import glob
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns

files = glob.glob("experiments/synthetic/*.csv")
files.sort()

df = pd.DataFrame(columns=["algorithm", "max_range", "real_cardinality", "estimated_cardinality", "ratio", "size", "insertion_time_mean", "insertion_time_std", "estimation_time"]) 
for file in files:
    df = pd.concat([df, pd.read_csv(file)], ignore_index=True)

df["method"] = df["algorithm"].apply(lambda x: x.split("_")[0])
df["p"] = df["algorithm"].apply(lambda x: x.split("_")[1] if "hll" in x else None)
df["hash"] = df["algorithm"].apply(lambda x: x.split("_")[2] if "hll" in x else None)
df["eps"] = df["algorithm"].apply(lambda x: x.split("_")[1] if "tbe" in x else None)
df["delta"] = df["algorithm"].apply(lambda x: x.split("_")[2] if "tbe" in x else None)

# Convert size in bytes to MB
df["size"] = df["size"].apply(lambda x: x / 1000000)

algorithm_mapping = {
    "hll_14_mmh3": r"HLL ($p=14$, hash=mmh3)",
    "hll_14_sha256": r"HLL ($p=14$, hash=SHA256)",
    "hll_16_mmh3": r"HLL ($p=16$, hash=mmh3)",
    "hll_16_sha256": r"HLL ($p=16$, hash=SHA256)",

    "hllpp_14_mmh3": r"HLL++ ($p=14$, hash=mmh3)",
    "hllpp_14_sha256": r"HLL++ ($p=14$, hash=SHA256)",
    "hllpp_16_mmh3": r"HLL++ ($p=16$, hash=mmh3)",
    "hllpp_16_sha256": r"HLL++ ($p=16$, hash=SHA256)",

    "tbe_0.1_0.05": r"TBE ($\epsilon=0.1, \delta=0.05$)",
    "tbe_0.05_0.05": r"TBE ($\epsilon=0.05, \delta=0.05$)",
}

df["number"] = df["algorithm"].apply(lambda x: np.where(np.array(sorted(list(algorithm_mapping.keys()))) == x)[0][0])
df["algorithm"].replace(algorithm_mapping, inplace=True)
df.sort_values(by=["algorithm"], inplace=True)

# # Ratio (per cardinality)
sns.set_theme(style="whitegrid")
sns.catplot(x="real_cardinality", y="ratio", hue="algorithm", data=df, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.05, join=False, dodge=.5)
plt.xlabel("Cardinality")
plt.ylabel("Ratio")
plt.savefig("experiments/synthetic/plots/ratio.pdf", dpi=800)
plt.close()

# Estimation time (per cardinality)
sns.set_theme(style="whitegrid")
sns.catplot(x="real_cardinality", y="estimation_time", hue="algorithm", data=df, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
plt.xlabel("Cardinality")
plt.ylabel("Estimation time (ms)")
plt.savefig("experiments/synthetic/plots/estimation_time.pdf", dpi=800)
plt.close()

# Insertion time (per cardinality)
sns.set_theme(style="whitegrid")
sns.catplot(x="real_cardinality", y="insertion_time_mean", hue="algorithm", data=df, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
plt.xlabel("Cardinality")
plt.ylabel("Avg. Insertion Time (ms)")
plt.savefig("experiments/synthetic/plots/insertion_time.pdf", dpi=800)
plt.close()

# Insertion time (per cardinality) - removing outlier
sns.set_theme(style="whitegrid")
sns.catplot(x="real_cardinality", y="insertion_time_mean", hue="algorithm", data=df[df.algorithm != algorithm_mapping["hll_16_sha256"]], kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
plt.xlabel("Cardinality")
plt.ylabel("Avg. Insertion Time (ms)")
plt.savefig("experiments/synthetic/plots/insertion_time-no-outlier.pdf", dpi=800)
plt.close()

# Memory usage barplot (per cardinality)
sns.set_theme(style="whitegrid")
sns.catplot(x="real_cardinality", y="size", hue="algorithm", data=df, kind="bar", height=5, aspect=2, estimator="mean", errorbar=None)
plt.xlabel("Cardinality")
plt.ylabel("Memory usage (MB)")
plt.savefig("experiments/synthetic/plots/memory_usage.pdf", dpi=800)
plt.close()

df_hll = deepcopy(df[df.method.str.contains("hll")])
df_hll["method"] = df_hll["method"].apply(lambda x: "HLL++" if x == "hllpp" else "HLL")

# # See difference in ratio per hash function and cardinality
sns.set_theme(style="whitegrid")
g = sns.catplot(x="real_cardinality", y="ratio", hue="hash", col="method", data=df_hll, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
g.set_xlabels("Cardinality")
g.set_ylabels("Ratio")
g.set_titles("{col_name}")
plt.savefig("experiments/synthetic/plots/ratio-hll.pdf", dpi=800)
plt.close()

# # See difference in insertion time per hash function and cardinality
sns.set_theme(style="whitegrid")
sns.catplot(x="real_cardinality", y="insertion_time_mean", hue="hash", data=df_hll, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
plt.xlabel("Cardinality")
plt.ylabel("Avg. Insertion Time (ms)")
plt.savefig("experiments/synthetic/plots/insertion_time-hll.pdf", dpi=800)
plt.close()