#!/bin/bash
python test_realtime.py \
    --device 'cpu' \
    --dataset 'SRD' \
    --samplepath './samples' \
    --cameraID 0 \
    --optimize \
    --prune \
    --quantize