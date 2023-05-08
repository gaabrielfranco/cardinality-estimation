import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

files = glob.glob("experiments/graph/*.csv")
files.sort()

df = pd.DataFrame(columns=["algorithm", "real_cardinality", "estimated_cardinality", "ratio", "size", "insertion_time_mean", "insertion_time_std", "estimation_time"]) 

for file in files:
    df_file = pd.read_csv(file, index_col=0)
    df_file.index.name = None
    df_file["algorithm"] = df_file.apply(lambda x: f"{x.algorithm}_{x.p}_{x.hash}_{x.n_hlls}" if "hll" in x["algorithm"] else f"{x.algorithm}_{x.eps}_{x.delta}", axis=1)
    df = pd.concat([df, df_file], ignore_index=True)

# Convert size in bytes to MB
df["size"] = df["size"].apply(lambda x: x / 1000000)

algorithm_mapping = {
    
    "hllpp_14_mmh3_10": "HLL++\n p=14\nhash=mmh3\nNH=10",
    "hllpp_14_mmh3_100": "HLL++\n p=14\nhash=mmh3\nNH=100",
    "hllpp_14_sha256_10": "HLL++\n p=14\nhash=SHA256\nNH=10",
    "hllpp_14_sha256_100": "HLL++\n p=14\nhash=SHA256\nNH=100",
    "hllpp_16_mmh3_10": "HLL++\n p=16\nhash=mmh3\nNH=10",
    "hllpp_16_mmh3_100": "HLL++\n p=16\nhash=mmh3\nNH=100",
    "hllpp_16_sha256_10": "HLL++\n p=16\nhash=SHA256\nNH=10",
    "hllpp_16_sha256_100": "HLL++\n p=16\nhash=SHA256\nNH=100",
    "hllpp_18_mmh3_10": "HLL++\n p=18\nhash=mmh3\nNH=10",
    "hllpp_18_mmh3_100": "HLL++\n p=18\nhash=mmh3\nNH=100",
    "hllpp_18_sha256_10": "HLL++\n p=18\nhash=SHA256\nNH=10",
    "hllpp_18_sha256_100": "HLL++\n p=18\nhash=SHA256\nNH=100",

    "tbe_0.05_0.05": "TBE\n" + r"$\epsilon=0.05$" + "\n" + r"$\delta=0.05$"
}

df["number"] = df["algorithm"].apply(lambda x: np.where(np.array(sorted(list(algorithm_mapping.keys()))) == x)[0][0])

df["algorithm"].replace(algorithm_mapping, inplace=True)
df.sort_values(by=["algorithm"], inplace=True)

df_hll = df[df["algorithm"].str.contains("HLL")]

# Ratio (general)
sns.set_theme(style="whitegrid")
sns.catplot(x="algorithm", y="ratio", data=df, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
plt.xticks(rotation=90)
plt.xlabel("Algorithm")
plt.ylabel("Ratio")
plt.savefig("experiments/graph/plots/ratio_general.pdf", bbox_inches="tight")
plt.close()

# # Ratio by n_hlls
sns.set_theme(style="whitegrid", font_scale=2)
g = sns.catplot(x="p", y="ratio", hue="hash", col="n_hlls", data=df_hll, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
g.set_xlabels("p")
g.set_ylabels("Ratio")
g.set_titles("Number of HLLs (NH): {col_name}")
plt.savefig("experiments/graph/plots/ratio_n_hlls.pdf", bbox_inches="tight")
plt.close()

# Memory (general)
sns.set_theme(style="whitegrid")
sns.catplot(x="algorithm", y="size", data=df, kind="bar", height=5, aspect=2, estimator="mean", errorbar=None)
plt.xticks(rotation=90)
plt.xlabel("Algorithm")
plt.ylabel("Size (MB)")
plt.savefig("experiments/graph/plots/size_general.pdf", bbox_inches="tight")
plt.close()

# Insertion time
sns.set_theme(style="whitegrid")
sns.catplot(x="algorithm", y="insertion_time_mean", data=df, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
plt.xticks(rotation=90)
plt.xlabel("Algorithm")
plt.ylabel("Insertion time (ms)")
plt.savefig("experiments/graph/plots/insertion_time.pdf", bbox_inches="tight")
plt.close()

# Estimation time
sns.set_theme(style="whitegrid")
sns.catplot(x="algorithm", y="estimation_time", data=df, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1, join=False)
plt.xticks(rotation=90)
plt.xlabel("Algorithm")
plt.ylabel("Estimation time (ms)")
plt.savefig("experiments/graph/plots/estimation_time.pdf", bbox_inches="tight")
plt.close()


