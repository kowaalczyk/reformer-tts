#!/bin/bash

cd /scidatalg
mkdir /scidatalg/reformer-tts
chmod a+rwx /scidatalg/reformer-tts
cd /scidatalg/reformer-tts

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda3

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

export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/gcp-cred.json"

git clone https://github.com/kowaalczyk/reformer-tts.git
cd reformer-tts
git checkout entropy-scripts

conda env create -f environment.yml
conda activate reformer-tts

dvc pull data_pipeline/lj_speech_tacotron2.dvc