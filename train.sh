#!/bin/bash
python train.py \
    --dataset 'SRD' \
    --datasetpath './dataset/SRD' \
    --iteration 20 \
    --batch_size 1 \
    --lr 0.0001 \
    --device 'cpu' \
    
    