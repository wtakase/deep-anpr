#!/bin/sh

REPORT_STEPS=500
#REPORT_STEPS=10
BATCH_SIZE=100
MAX_STEPS=25000
#MAX_STEPS=10

#DROPOUT_RATIOS="0 0.25 0.5 0.75"
DROPOUT_RATIO=0

LEARN_RATES="0.001 0.01 0.1"

#for DROPOUT_RATIO in $DROPOUT_RATIOS; do
for LEARN_RATE in $LEARN_RATES; do
    datetime=`date +"%Y%m%d%H%M%S"`
    output_file="weights/weights_${datetime}.npz"
    ./train.py -r $REPORT_STEPS -b $BATCH_SIZE -m $MAX_STEPS -d $DROPOUT_RATIO -l $LEARN_RATE -o $output_file > ./logs/train_${datetime}.log 2>&1
done
