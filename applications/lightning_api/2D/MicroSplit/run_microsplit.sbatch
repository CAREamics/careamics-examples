#!/bin/bash

#SBATCH --job-name=MicroSplit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=igor.zubarev@fht.org
#SBATCH --partition=dgx
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --chdir=/home/igor.zubarev/projects/careamics-examples/applications/lightning_api/2D/MicroSplit
#SBATCH --error=/home/igor.zubarev/projects/careamics-examples/applications/lightning_api/2D/MicroSplit/logs/error_%j.log
#SBATCH --output=/home/igor.zubarev/projects/careamics-examples/applications/lightning_api/2D/MicroSplit/logs/output_%j.log

source ~/.bashrc
conda activate cmcs

mkdir -p logs

python microsplit_lightning_api.py \
    --pred-data ./data/pred_data \
    --highsnr-data ./highSNR_test_data.tiff \
    --output-metrics ./results/metrics.json


