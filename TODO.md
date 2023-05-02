~~space~~
~~insert time/item~~
~~querying time~~
~~accuracy (ratio)~~ 

~~use different hash functions~~

Datasets:
https://www.gutenberg.org/

https://commoncrawl.org/2017/05/hostgraph-2017-feb-mar-apr-crawls/:
So each item let’s say is a tuple of the form (u,v)
Count the number of distinct edges in a multigraph
Or an even simpler would be to count the number of nodes

Total # of IPs

Another thing you want to do, is to produce plots of the above 1.2.3.4. versus the number of counters
P=16 or for huge counts P=18

https://arxiv.org/abs/2208.10578

~~Metric: ratio (and I can see if they are overestimating or underestimating)~~

Third party software:

Apache streaming (for distributed working)
C/C++ implementations (http://dialtr.github.io/libcount/)

Experimental setup:

Datasets:
    - Replicate paper "datasets" with different cardinalities
        - 1000, 10000, 20000, 40000, 60000, 80000

    - Sample of the number of edges in the graph

HLL/HLL++ params:
    p=[10, 14, 16, 18 (HLL++ only)]
    hash="mmh3", "sha256"

Textbook params:
    eps, delta =[(0.1, 0.1), (0.05, 0.05)]

Measure:
    - space
    - insert time/item
    - querying time
    - accuracy (ratio)

