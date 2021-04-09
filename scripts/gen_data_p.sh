#!/bin/sh
set -eux

DATA=$(pwd)/../data

echo "data dir is: " $DATA

cd $(pwd)/../extern/Graph-WaveNet

# PEMS-BAY
python generate_training_data.py --output_dir=$DATA/PEMA-BAY --traffic_df_filename=$DATA/raw/pems-bay.h5
