#!/bin/bash

# you may want to adjust some of options below
#SBATCH --time=2-23:00:00
#SBATCH --job-name=trainExample
#SBATCH --output=/results/reformer-tts/trainExample.out
#SBATCH --error=/results/reformer-tts/trainExample.err
#SBATCH --nodelist=asusgpu1,sylvester,arnold


#SBATCH --qos=2gpu3d
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --export=ALL,PATH="/scidatalg/reformer-tts/miniconda3/bin:${PATH}"
#SBATCH --signal=SIGUSR1@90  # for auto resubmit

# debugging flags (optional) - suggested in pytorch lightining docs
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

__conda_setup="$('/scidatalg/reformer-tts/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
eval "$__conda_setup"
else
if [ -f "/scidatalg/reformer-tts/miniconda3/etc/profile.d/conda.sh" ]; then
. "/scidatalg/reformer-tts/miniconda3/etc/profile.d/conda.sh"
else
export PATH="/scidatalg/reformer-tts/miniconda3/bin:$PATH"
fi
fi
unset __conda_setup
cd /scidatalg/reformer-tts/reformer-tts || exit 1
conda activate reformer-tts

conda env update -f environment.yml
pip install -e .

echo "Visible devices: $CUDA_VISIBLE_DEVICES"
echo "Device load:"
/usr/bin/nvidia-smi

# HERE change command here
python -O -m reformer_tts.cli \
  --config /scidatalg/reformer-tts/reformer-tts/config/entrpy-reformer-lj.yml \
  train-tts \
  --resume '/results/reformer-tts/reformer-tts_ljtc2_double-loss_entropy/epoch=10-val_loss=0.32.ckpt'
