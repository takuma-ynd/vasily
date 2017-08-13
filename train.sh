#!/bin/bash

echo "log file path?(default:tune.log): "
DEFAULT_LOG_FILE="tune.log"
read LOG_FILE
if [[ $LOG_FILE == "" ]]
then
    LOG_FILE=$DEFAULT_LOG_FILE
fi

# array=(sth0 sth1 sth2) <-- bashにおける配列の定義方法
f_list=(50 60 70 80 90 100)
a_list=(20 30 40 50 60)
lmbda_list=(0.000001 0.00001 0.0001 0.001)
num_epochs_list=(5 10 15)
COUNT=0
# ${array[@]} --> (sth0 sth1 sth2)
# $array --> sth0

for f in ${f_list[@]}; do
    for a in ${a_list[@]}; do
    	for lmbda in ${lmbda_list[@]}; do
    	    for num_epochs in ${num_epochs_list[@]}; do
		COUNT=$(( COUNT + 1 ))
		echo "----------TRAINING----------"
    		echo "f:$f"
    		echo "a:$a"
    		echo "lmbda:$lmbda"
    		echo "num_epochs:$num_epochs"
		echo ""
		python3 main.py --train --train-file ml-100k/u1.base -f $f -a $a --lmbda $lmbda --num-epochs $num_epochs --model-file u1_id$COUNT.pickle --eval-file ml-100k/u1.test --log-file $LOG_FILE

    	    done
    	done
    done
done

