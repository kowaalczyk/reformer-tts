#!/bin/bash


#SBATCH --job-name=asusgpu3_test
#SBATCH --qos=gsn
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --output=/results/reformer-tts/asusgpu3.out.txt
#SBATCH --export=ALL,PATH="/scidatalg/reformer-tts/miniconda3/bin:${PATH}"

#SBATCH --nodelist=<node_name> # HERE change node name here

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