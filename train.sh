#!/bin/bash
python train.py \
    --dataset 'SRD' \
    --datasetpath '/home/luni/shadow_removal_project/dataset/SRD' \
    --iteration 20 \
    --batch_size 1 \
    --lr 0.0001 \
    --device 'cpu' \
    
    