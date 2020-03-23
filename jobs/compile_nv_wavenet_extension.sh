#!/bin/bash
#
#SBATCH --job-name=compile_nv_wavenet_extension
#SBATCH --partition=common
#SBATCH --qos=gsn
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --export=ALL

echo "=== $(date) === debug ==="
which python
python --version
cd "/scidatasm/$USER/nv_wavenet/pytorch/" || exit

echo "=== $(date) === compilation ==="
make || exit

echo "=== $(date) === build ==="
python build.py build || exit

echo "=== $(date) === install ==="
python build.py install || exit

echo "=== $(date) === test ==="
python nv_wavenet_test.py || exit

echo "=== $(date) === copying results ==="
cp "./libwavenet_infer.so" "/results/$USER/libwavenet_infer.so"

mkdir "/results/$USER/build/" || echo "skipped directory creation: build"
cp -r ./build/* "/results/$USER/build/"

mkdir "/results/$USER/wavenet-test/" || echo "skipped directory creation: wavenet-test"
cp ./*.wav "/results/$USER/wavenet-test/"

echo "=== $(date) === done ==="
