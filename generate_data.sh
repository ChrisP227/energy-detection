#!/bin/bash

YAML=dataset.yaml

# Generating training dataset
ROOT=dataset/train
NUMSPEC=1000
SEED=13

python create_dataset.py $ROOT -y $YAML -n $NUMSPEC -s $SEED --multithread

# Generate validation dataset
ROOT=dataset/validate
NUMSPEC=100
SEED=42

python create_dataset.py $ROOT -y $YAML -n $NUMSPEC -s $SEED --multithread