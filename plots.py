import pandas as pd
import glob
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns

files = glob.glob("experiments/synthetic/*.csv")

# removing hll-cython
files = [f for f in files if "hll-cython" not in f]
files.sort()

df = pd.DataFrame(columns=["algorithm", "max_range", "real_cardinality", "estimated_cardinality", "ratio", "size", "insertion_time_mean", "insertion_time_std", "estimation_time"]) 
for file in files:
    df = pd.concat([df, pd.read_csv(file)], ignore_index=True)

df["method"] = df["algorithm"].apply(lambda x: x.split("_")[0])
df["p"] = df["algorithm"].apply(lambda x: x.split("_")[1] if "hll" in x else None)
df["hash"] = df["algorithm"].apply(lambda x: x.split("_")[2] if "hll" in x else None)
df["eps"] = df["algorithm"].apply(lambda x: x.split("_")[1] if "tbe" in x else None)
df["delta"] = df["algorithm"].apply(lambda x: x.split("_")[2] if "tbe" in x else None)

# Distribution of ratios
# ecdf_tbe = ECDF(df[df.method == "tbe"].ratio.values)
# ecdf_hllpp = ECDF(df[df.method == "hllpp"].ratio.values)
# ecdf_hll = ECDF(df[df.method == "hll"].ratio.values)

# sns.set_theme(style="whitegrid")
# plt.plot(ecdf_tbe.x, ecdf_tbe.y, label="TBE")
# plt.plot(ecdf_hllpp.x, ecdf_hllpp.y, label="HLL++")
# plt.plot(ecdf_hll.x, ecdf_hll.y, label="HLL")
# plt.legend()
# plt.xlabel("Ratio")
# plt.ylabel("ECDF")
# plt.show()

# Distribution of average insertion time
# ecdf_tbe = ECDF(df[df.method == "tbe"].insertion_time_mean.values)
# ecdf_hllpp = ECDF(df[df.method == "hllpp"].insertion_time_mean.values)
# ecdf_hll = ECDF(df[df.method == "hll"].insertion_time_mean.values)

# sns.set_theme(style="whitegrid")
# plt.plot(ecdf_tbe.x, ecdf_tbe.y, label="TBE")
# plt.plot(ecdf_hllpp.x, ecdf_hllpp.y, label="HLL++")
# plt.plot(ecdf_hll.x, ecdf_hll.y, label="HLL")
# plt.legend()
# plt.xlabel("Average insertion time (ms)")
# plt.ylabel("ECDF")
# plt.show()

df["real_cardinality"] = df["real_cardinality"].astype(int)

# # Distribution of ratios (different plot per cardinality)
# sns.set_theme(style="whitegrid")
# fig, ax =  plt.subplots(2, 4, figsize=(10, 5), sharey=True, sharex=True)
# i, j = 0, 0
# for idx, real_cardinality in enumerate(sorted(df.real_cardinality.unique())):
#     ecdf_tbe = ECDF(df[(df.method == "tbe") & (df.real_cardinality == real_cardinality)].ratio.values)
#     ecdf_hllpp = ECDF(df[(df.method == "hllpp") & (df.real_cardinality == real_cardinality)].ratio.values)
#     ecdf_hll = ECDF(df[(df.method == "hll") & (df.real_cardinality == real_cardinality)].ratio.values)
#     ax[i, j].plot(ecdf_tbe.x, ecdf_tbe.y, label="TBE")
#     ax[i, j].plot(ecdf_hllpp.x, ecdf_hllpp.y, label="HLL++")
#     ax[i, j].plot(ecdf_hll.x, ecdf_hll.y, label="HLL")
#     if i == 1:
#         ax[i, j].set_xlabel("Ratio")
#     if j == 0:
#         ax[i, j].set_ylabel("ECDF")
#     ax[i, j].set_title(f"Cardinality: {real_cardinality}")
#     if j == 3:
#         i += 1
#         j = 0
#     else:
#         j += 1

# ax[-1, -1].axis('off')
# plt.tight_layout()
# # Making space for legend
# #plt.subplots_adjust(right=0.94)
# # Adding legend
# ax[0, 3].legend(bbox_to_anchor=(0.7, -0.5), loc='upper right', borderaxespad=0., title="Algorithm")
# #plt.show()
# plt.savefig("experiments/synthetic/plots/ratio.png", dpi=1200)
# plt.close()

# # # Distribution of average insertion time (different plot per cardinality)
# sns.set_theme(style="whitegrid")
# fig, ax =  plt.subplots(1, 7, figsize=(20, 5), sharey=True, sharex=True)
# for idx, real_cardinality in enumerate(sorted(df.real_cardinality.unique())):
#     ecdf_tbe = ECDF(df[(df.method == "tbe") & (df.real_cardinality == real_cardinality)].insertion_time_mean.values)
#     ecdf_hllpp = ECDF(df[(df.method == "hllpp") & (df.real_cardinality == real_cardinality)].insertion_time_mean.values)
#     ecdf_hll = ECDF(df[(df.method == "hll") & (df.real_cardinality == real_cardinality)].insertion_time_mean.values)
#     ax[idx].plot(ecdf_tbe.x, ecdf_tbe.y, label="TBE")
#     ax[idx].plot(ecdf_hllpp.x, ecdf_hllpp.y, label="HLL++")
#     ax[idx].plot(ecdf_hll.x, ecdf_hll.y, label="HLL")
#     ax[idx].set_xlabel("Average insertion time (ms)")
#     if idx == 0:
#         ax[idx].set_ylabel("ECDF")
#     ax[idx].set_title(f"Cardinality: {real_cardinality}")


# plt.tight_layout()
# # Making space for legend
# plt.subplots_adjust(right=0.8)
# # Adding legend
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
# plt.show()

# Estimation time (different plot per cardinality)
# sns.set_theme(style="whitegrid")
# sns.catplot(x="real_cardinality", y="estimation_time", hue="method", data=df, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1)
# plt.xlabel("Cardinality")
# plt.ylabel("Estimation time (ms)")
# plt.savefig("experiments/synthetic/plots/estimation_time.png", dpi=800)

# Insertion time (different plot per cardinality)
sns.set_theme(style="whitegrid")
sns.catplot(x="real_cardinality", y="insertion_time_mean", hue="method", data=df, kind="point", height=5, aspect=2, estimator="mean", errorbar=("ci", 95), capsize=0.1)
plt.xlabel("Cardinality")
plt.ylabel("Avg. Insertion Time (ms)")
plt.savefig("experiments/synthetic/plots/insertion_time.png", dpi=800)


# Convert size in bytes to MB
df["size"] = df["size"].apply(lambda x: x / 1000000)

# # Memory usage barplot (per cardinality)
# sns.set_theme(style="whitegrid")
# sns.catplot(x="real_cardinality", y="size", hue="method", data=df, kind="bar", height=5, aspect=2, estimator="mean", errorbar=None)
# plt.xlabel("Cardinality")
# plt.ylabel("Memory usage (MB)")
# plt.savefig("experiments/synthetic/plots/memory_usage.png", dpi=800)







