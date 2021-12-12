#!/bin/bash

# Dataset name
DATASET="data-2 data-30"

# Default experimental parameters in competition
# Number of iterations
# To run faster, you can set N_ITERATION smaller in local test
N_ITERATION=20
# number of suggestions
N_SUGGESTION=5
# number of repeats
N_REPEAT=10
echo N_REPEAT $N_REPEAT

# Test random search
# Directory of searcher.py
# SEARCHER="example_random_searcher"

# Run searcher in all dataset
# python main.py -o $SEARCHER -d $DATASET -i $N_ITERATION -s $N_SUGGESTION -r $N_REPEAT

# Test bayesian optimization
# This searcher costs about 10 minutes.  
# SEARCHER="EGO_LHS"======upload_EGO_LHS30_PseudoEI_MP_std_08240105
SEARCHER="PEI_MP75_std_60_LHS20r_08272204"
#upload_PEI_MP75_std-60_LHS20r_08272204-test
echo $SEARCHER
python main.py -o $SEARCHER -d $DATASET -i $N_ITERATION -s $N_SUGGESTION -r $N_REPEAT
# python main.py -o "example_bayesian_optimization" -d "data-2 data-30" -i 100 -s 1 -r 30

# Run searcher in one dataset for one repeat
# ONE_DATASET="data-2"
# python ./thpo/run_search_one_time.py -o $SEARCHER -d $ONE_DATASET -i $N_ITERATION -s $N_SUGGESTION -n 1
#Pycharm: -o "EGO_LHS" -d "data-30" -i 20 -s 5 -r 2