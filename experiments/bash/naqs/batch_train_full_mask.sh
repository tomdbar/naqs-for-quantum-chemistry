#!/bin/bash

GPU_NUM=$1
MOLECULE_NAME=$2
MOLECULE_NAME_SAFE=$(echo $MOLECULE_NAME | tr '/' '_')

echo "GPU_NUM  = ${GPU_NUM}"
echo "MOLECULE NAME  = ${MOLECULE_NAME}"

run=1
for seed in {111,222,333,444,555}
do
    echo "running exp ${run}/5...output can be viewed at ${MOLECULE_NAME_SAFE}_s${seed}.out"
    CUDA_VISIBLE_DEVICES=${GPU_NUM} nohup python -u -m experiments.run -o "data/fullMask/${MOLECULE_NAME}_s${seed}" -m "molecules/${MOLECULE_NAME}" -single_phase -n1 -n_layer 1 -n_hid 64 -n_layer_phase 2 -n_hid_phase 512 -s ${seed} -n_train 10000 -output_freq 25 -save_freq -1 -full_mask_psi >&1 | tee outfile>${MOLECULE_NAME_SAFE}_s${seed}.out
    ((run++))
done