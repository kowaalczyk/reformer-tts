#!/bin/bash


#SBATCH --job-name=reformerNoise
#SBATCH --qos=gsn
#SBATCH --partition=common
#SBATCH --export=ALL,PATH="/scidatalg/reformer-tts/miniconda3/bin:${PATH}"
#SBATCH --signal=SIGUSR1@90  # to auto resubmit job


#SBATCH --gres=gpu:titanx:1  # change this if you want to train on less gpus
#SBATCH --ntasks-per-node=1  # change this if you want to train on less gpus

#SBATCH --error=/results/reformer-tts/reformerNoise.err
#SBATCH --output=/results/reformer-tts/reformerNoise.out  # HERE you may want to change this
#SBATCH --nodelist=asusgpu4 # HERE change node name here

cd /scidatalg/reformer-tts/reformer-tts
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

conda activate reformer-tts

# HERE change command here
python -m reformer_tts.cli --config /scidatalg/reformer-tts/reformer-tts/config/entrpy-reformer-lj.yml train-tts