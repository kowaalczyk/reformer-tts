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

### 1. Install dependencies

#### Using conda

Thanks to conda-forge community, we can install all packages (including necessary
binaries, like `ffmpeg`) using one command.

```shell script
conda env create -f environment.yml
```


#### Using other package managers

1. Check your environment and ensure you have `Python>=3.8`:
```shell
which python
python --version
```

2. Install python dependencies (also installs our package in editable mode):
```shell
pip install -r requirements.txt
```

3. Ensure you have `ffmpeg>=3.4,<4.0` installed ([installation instructions](https://www.ffmpeg.org/download.html))


### 2. Configure tools

1. In order for dvc to have write access to the remote, 
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

2. Get all of the data - this step needs to be repeated:
    - every time you start working after a break
    - after every git pull
    - after checking out another git branch
```shell
dvc pull
```


### Setup details

- Use whatever package manager you want
- Use `Python>=3.8`
- All python dependencies will be in `requirements.txt` 
  as well as in `environment.yml`


### Configuration

Configuration is organized in dataclass structures: 
- Each project submodule has its own configuration file, called `config.py`, 
  where the parameters and *default* values are defined - for example,
  dataset config parameters are specified in `reformer_tts.dataset.config`
- The `reformer_tts.config.Config` class contains all submodules' config settings
- *Actual* values of config parameters are loaded from configuration files, which
  define an instance of `Config` class that is loaded by CLI. For convenience,
  and support for python types, these config files are just plain python files.

This way, the default values are set close to the place where they are used,
any config value can be overridden wherever you want, and we take advantage
of python typing to eliminate unnecessary conversions from strings yaml / json. 

**To change runtime configuration**
- create a python file, which imports `reformer_tts.Config`
- in this file, create `CONFIG` variable, set it to an instance of `Config` 
  class with its fields set to the values you want 
- tell CLI to use your config:
    - using environment variable: `export REFORMER_TTS_CONFIG=/path/to/your_config.py`
    - command line option: `python reformer_tts/cli.py --config=/path/to/your_config.py`

**To add configuration for new module**
- create `config.py` in your module
- define a dataclass with all necessary config parameters in the new file:
    - make sure your class does not re-define parameter values for other config files
      (ie. we specified number of spectrogram channels only once - in the same place
      for both `dataset` and `squeezewave` modules)
    - make sure your class has default values for all the parameters
- add field for your dataclass in the `reformer_tts.config` main config class


### Data dependencies

We use [DVC](https://dvc.org/) for defining data processing pipelines,
with remote set to `s3://reformer-tts/dvc`.

Credentials for DVC have access to entire `reformer-tts` bucket, so that we can 
use it to create other folders (s3 prefixes) for releasing models, etc.


### Setup for running jobs on [entropy cluster](entropy.mimuw.edu.pl)

Job definition files are located in [`jobs/`](jobs) directory.

File `setup_jobs.sh` was created to help with setting up environment for jobs:
```
./setup_jobs.sh help

Setup tasks:
./setup_jobs.sh dirs - make directories necessary to run the jobs
./setup_jobs.sh sync - sync all necessary data to /scidatasm/$USER/ partiion
./setup_jobs.sh clean_users - change usernames in job files to a generic $USER
./setup_jobs.sh all - perform all of the setup tasks in sequence

Running jobs:
./setup_jobs.sh check - checks scripts for common errors
./setup_jobs.sh run [job_file] performs checks and runs the job using sbatch
```

Running jobs manually may result in errors or data loss.
To prevent it, use `./setup_jobs.sh run [job_file]` instead of `sbatch` directly.

Example:
```shell script
./setup_jobs.sh run jobs/compile_nv_wavenet_extension.sh
```

This will automatically save job output with its name and timestamp in your results folder.


#### Adding new jobs

Before sharing your job file with others, document what changes need to be made
in the job file, so that it works for other users. Make sure to include:
1. Changes to user-specific paths (possibly requires changing `setup_jobs.sh),
   as #SBATCH directives cannot use environment variables
   (see [related docs](https://help.rc.ufl.edu/doc/Using_Variables_in_SLURM_Jobs))
2. Directories that need to be created (otherwise the script will crash)
3. Results that need to be moved (jobs save results in /results/ partitions,
   usually we'll want to add results to dvc or some other local path)


### TODOs

- configure neptune for experiment tracking
- document hardware specs (GPU) as soon as we set up training
