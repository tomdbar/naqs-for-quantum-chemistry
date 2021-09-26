#!/bin/bash

GPU_NUM=$1

for mol in {H2,F2,HCl,LiH,H2O,BeH2,H2S,NH3}
do
    experiments/bash/naqs/batch_train_full_mask_no_amp_sym.sh $GPU_NUM ${mol}
done

