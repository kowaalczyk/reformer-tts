#!/bin/bash

# you may want to adjust some of options below
#SBATCH --time=2-23:00:00
#SBATCH --job-name=reformer-clipping-dropout
#SBATCH --output=/results/reformer-tts/reformer-clipping-dropout.out
#SBATCH --error=/results/reformer-tts/reformer-clipping-dropout.err
#SBATCH --constraint=homedir
#SBATCH --nodelist=arnold

#SBATCH --qos=2gpu3d
#SBATCH --partition=common
#SBATCH --gres=gpu:1

# debugging flags (optional) - suggested in pytorch lightining docs
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /home/kk385830/reformer-tts || exit 1
conda activate reformer-tts

conda env update -f environment.yml
pip install -e .

echo "Visible devices: $CUDA_VISIBLE_DEVICES"
echo "Device load:"
/usr/bin/nvidia-smi

# HERE change command here
python -O -m reformer_tts.cli \
  --config /home/kk385830/reformer-tts/config/entropy-reformer-lj-dropout-clipping.yml \
  train-tts
