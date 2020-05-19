#!/bin/bash

#SBATCH --job-name=dvc_pull
#SBATCH --qos=gsn
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --export=ALL,PATH="/scidatalg/reformer-tts/miniconda3/bin:${PATH}"

#SBATCH --nodelist=<node_name> # HERE change node name here
#SBATCH --output=/results/reformer-tts/dvc_pull.txt   # HERE you may want to change this also

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
export GOOGLE_APPLICATION_CREDENTIALS="/scidatalg/reformer-tts/gcp-cred.json"

# HERE change command here
dvc pull data_pipeline/lj_speech_tacotron2.dvc