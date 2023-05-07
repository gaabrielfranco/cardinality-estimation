for algorithm in "hllpp"
do
    for p in "14" "16" "18"
    do
        for hash in "mmh3" "sha256"
        do
            for n_hlls in "10" "100"
            do
                for ((exec=0;exec<10;exec++))
                do
                    python3 graph_experiments.py -a $algorithm -p $p -hs $hash -n $n_hlls -ne $exec -s 0 -e 100000000
                done
            done
        done
    done
done 

for algorithm in "tbe"
do
    for eps in "0.05"
    do
        for delta in "0.05"
        do
            for ((exec=0;exec<10;exec++))
            do
                python3 graph_experiments.py -a $algorithm -eps $eps -d $delta -ne $exec -s 0 -e 100000000
            done
        done
    done
done 