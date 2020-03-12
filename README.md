# Reformer-TTS

An adaptation of [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) 
for text-to-speech task.


## Project scope

We aim to create a significantly more efficient version of state-of-the-art text-to-speech model, by replacing its transformer architecture with optimizations proposed in the more recent reformer paper. We’ll use it to generate a believable deepfake of Donald Trump based on a custom dataset of his speeches created specifically for this purpose.

Specifically, we want to:

1. Use layers introduced in Reformer ([paper](https://arxiv.org/abs/2001.04451), [original implementation](https://github.com/google/trax/tree/master/trax/models/reformer), [pytorch implementation](https://github.com/lucidrains/reformer-pytorch)) to build a **Reformer-TTS** model inspired by **Transformer-TTS** ([paper](https://arxiv.org/abs/1809.08895), [unreliable implementation in pytorch](https://github.com/soobinseo/Transformer-TTS)).
2. Build a custom **Trump Speech Dataset** based on scraped videos with transcripts ([source](https://www.rev.com/blog/transcript-tag/donald-trump-speech-transcripts))
3. Train the Reformer-TTS on the created dataset, evaluate results using objective metrics (from [this paper](https://arxiv.org/abs/1909.11646)) and implement a pipeline for human evaulation (MOS) using WaveNet vocoder ([paper](https://arxiv.org/abs/1609.03499), [NVIDIA reference implementation](https://github.com/NVIDIA/nv-wavenet)) to synthesize voice from mel spectrograms
4. Write a paper summarizing our results

Our deliverable will include:

- ready-to-use model with pre-trained weights (all publicly available)
- pipeline for training reproduction (from data download to trained model)
- paper & presentation describing our decisions & results


### Extra documents

- [project journal](https://paper.dropbox.com/doc/GSN-2020-Transformer-Project-Journal--Av9TZdQgTjFBPDsh~F_GD4uRAQ-Y2zXcN0nSKlmMYPjLTzMw)
- [research doc](https://paper.dropbox.com/doc/GSN-2020-Speech-Synthesis-Research-Doc--Av8RCqsp~MX95ZSt3Jl1ubgSAQ-Iv6r0eA0nmS34RYK8BCmK)


## Development setup

1. Check your environment and ensure you have `Python>=3.8`:
```shell
which python
python --version
```

2. Install python dependencies:
```shell
pip install -r requirements.txt
```

3. In order for dvc to have write access to the remote, 
configure your aws account (using credentials from the csv):
```shell
aws configure
# Copy from provided CSV:
# - AWS Access Key ID
# - AWS Secret Access Key
# 
# Specify region manually:
# - Default region name: eu-west-1
#
# Leave blank:
# - Default output format
```
*NOTE: if you only need read acces (for reproduction), all you need is to specify the region*

**Steps 1-3 are performed only once, steps 4+ should be performed every time you start working after a break**

4. Get all of the data
```shell
dvc pull
```

5. Install `reformer_tts` package in editable format (auto-reload after file save)
```shell
pip install -e .
```

### Setup details

#### Environment and libraries

- Use whatever package manager you want
- Use `Python>=3.8`
- All python dependencies will be in `requirements.txt`


#### Data dependencies

We use [DVC](https://dvc.org/) for defining data processing pipelines,
with remote set to `s3://reformer-tts/dvc`.

Credentials for DVC have access to entire `reformer-tts` bucket, so that we can 
use it to create other folders (s3 prefixes) for releasing models, etc.


### TODOs

- setup workflow for running instructions on slurm cluster (for GPUs)
- configure neptune for experiment tracking
