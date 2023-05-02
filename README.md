# cardinality-estimation
Implementations of algorithms that estimate the cardinality of the data in a streaming fashion with insertions only.

## Dependencies

Python 3.8.16. To install the libraries needed, use the following command:

```bash
pip3 install -r requirements.txt
```

## Usage

```py
import numpy as np
from cardinality_estimation import HyperLogLog, HyperLogLogPlusPlus, LinearCounting, TextBookEstimation

# Constants
SEED = 326178
random = np.random.RandomState(SEED)

# Random data
D = random.choice(range(50000), size=1000000, replace=True)

# Estimators
hll = HyperLogLog(p=16, hash="sha256", random_state=SEED)
hllpp = HyperLogLogPlusPlus(p=18, hash="mmh3", random_state=SEED)
tbe = TextBookEstimation(eps=0.1, delta=0.1, m=len(D), random_state=SEED)
lc = LinearCounting(k=10000, hash="mmh3", random_state=SEED)

# Insertion
for x in D:
    hll.insert(x)
    hllpp.insert(x)
    tbe.insert(x)
    lc.insert(x)

# Estimation
print("Estimation (HLL): %d" % hll.get_estimate())
print("Estimation (HLL++): %d" % hllpp.get_estimate())
print("Estimation (TBE): %d" % tbe.get_estimate())
print("Estimation (LC): %d" % lc.get_estimate())
print("Real: %d" % len(set(D)))
```

## Replicating the experiments

### Synthetic experiments

To run the experiments, run the following command:

```sh
./run_synthetic_experiments.sh
```

### Common Crawl’s Web Graph experiment

To get the data, run the following command:

```sh
wget http://data.commoncrawl.org/projects/hyperlinkgraph/cc-main-2017-feb-mar-apr-hostgraph/edges.txt.gz
```

Then, unzip the file edges.txt.gz to get the edges.txt file.

To run the experiments, run the following command:

```sh
./run_graph_experiments.sh
```

### Plots

TODO

## License

[MIT License](LICENSE)

## Author

[Gabriel Franco](https://cs-people.bu.edu/gvfranco/)