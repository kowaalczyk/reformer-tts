# !/bin/bash

#SBATCH --job-name=reformer-tts-test
#SBATCH --qos=gsn
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --nodelist=asusgpu4

cd /scidatalg/reformer-tts/reformer-tts
conda activate reformer-tts
python -m reformer_tts.cli --config /scidatalg/reformer-tts/reformer-tts/config/entrpy-reformer-lj.yml train-tts