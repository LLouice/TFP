#!/bin/sh
set -eux

DATA=$(pwd)/../data

echo "data dir is: " $DATA

cd $(pwd)/../extern/Graph-WaveNet

python generate_training_data.py --output_dir=$DATA/METR-LA --traffic_df_filename=$DATA/raw/metr-la.h5

# # PEMS-BAY
# python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
