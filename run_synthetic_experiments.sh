# # HLL/HLL++
for max_range in "1000" "10000" "20000" "40000" "60000" "80000" "100000"
do
    for algorithm in "hllpp" "hll"
    do
        for p in "14" "16"
        do
            for hash in "mmh3" "sha256"
            do
                python3 synthetic_experiments.py -mr $max_range -a $algorithm -p $p -hs $hash
            done
        done
    done 
done

for max_range in "1000" "10000" "20000" "40000" "60000" "80000" "100000"
do
    for algorithm in "tbe"
    do
        for eps in "0.1" "0.05"
        do
            for delta in "0.05"
            do
                python3 synthetic_experiments.py -mr $max_range -a $algorithm -eps $eps -d $delta
            done
        done
    done 
done