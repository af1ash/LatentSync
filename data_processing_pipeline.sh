#!/bin/bash

python -m preprocess.data_processing_pipeline \
    --total_num_workers 1 \
    --per_gpu_num_workers 1 \
    --resolution 256 \
    --sync_conf_threshold 3 \
    --temp_dir /home/lxb/workspace/LatentSync/data/xiaowei2/temp \
    --input_dir /home/lxb/workspace/LatentSync/data/xiaowei2/raw
