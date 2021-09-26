#!/bin/bash

GPU_NUM=$1

for mol in {H2_6-31G, H2_cc-pvdz, H2_cc-pvtz, H2O_6-31G}
do
    experiments/bash/naqs/batch_train_full_mask.sh $GPU_NUM ${mol}
done