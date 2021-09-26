#!/bin/bash

GPU_NUM=$1

for mol in {H2,LiH,NH3,H2O,N2,C2,H2O_6-31G}
do
    experiments/bash/naqs/batch_train_full_mask.sh $GPU_NUM carleo/${mol}
done