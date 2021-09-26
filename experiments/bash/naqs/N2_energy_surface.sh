#!/bin/bash

GPU_NUM=$1

for r in {0.75,0.9,1.05,1.2,1.35,1.5,1.65,1.8,1.95,2.1,2.25}
do
    experiments/bash/naqs/batch_train_full_mask.sh $GPU_NUM N2_${r}
done