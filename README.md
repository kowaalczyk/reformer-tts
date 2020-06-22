# Reformer-TTS

An adaptation of [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)
for text-to-speech task.

This project contains:
- preprocessing code for creating a Trump Speech Dataset based on transcripts
  from [rev.com](https://www.rev.com/blog/transcript-tag/donald-trump-speech-transcripts)
- implementation of Reformer TTS: an adaptation of
  [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) for text-to-speech task,
  based on [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)
- implementation of [Squeezewave: Extremely Lightweight Vocoders For On-Device Speech Synthesis](https://arxiv.org/pdf/2001.05685)
  in modern PyTorch, without dependencies on Tacotron2, WaveNet or WaveGlow
- Pytorch Lightning wrappers for easy training of both models with easy-to-use configuration management
- CLI for running training, inference and data preprocessing


## Project scope and current status

We aimed to create a significantly more efficient version of state-of-the-art text-to-speech model,
by replacing its transformer architecture with optimizations proposed in the more recent reformer paper.
We’ll use it to generate a believable deepfake of Donald Trump based on a custom dataset of his speeches,
created specifically for this purpose.

Unfortunately, we weren't able to produce results matching the ones from Transformer TTS paper,
after experimenting with more than 100 hyperparameter combinations over 2 months. We believe that
the model size is a significant factor here, and to train transformers for TTS one really needs
to reduce overfitting to allow long, steady training process (~1 week of training on RTX 2080Ti).

Also, having access to original implementation of Transformer TTS would greatly help.

While the reformer didn't match our expectations, the SqueezeWave implementation matches performance of
[the original one](https://github.com/tianrengao/SqueezeWave) without FP16 support.

We also include CLI for running training and inference (see *usage* section),
and all data necessary for reproduction of experiments (see *development* section).

**The project is under a significant refactor, this version is left here to allow compatiblility
with our previous expeirments and will be moved in the near future**.


### Extra documents

- [final presentation](https://youtu.be/ckeKsM6obnM)
  and [slides](https://speakerdeck.com/kowaalczyk/reformer-text-to-speech)
- [project journal](https://paper.dropbox.com/doc/GSN-2020-Transformer-Project-Journal--Av9TZdQgTjFBPDsh~F_GD4uRAQ-Y2zXcN0nSKlmMYPjLTzMw)
- [research doc](https://paper.dropbox.com/doc/GSN-2020-Speech-Synthesis-Research-Doc--Av8RCqsp~MX95ZSt3Jl1ubgSAQ-Iv6r0eA0nmS34RYK8BCmK)


## Using the project

This project is a normal python package, and can be installed using `pip`,
as long as you have **Python 3.8 or greater**.

Go to [releases page](https://github.com/kowaalczyk/reformer-tts/releases)
to find the installation instruction for latest release.

After installation, you can see available commands by running:
```shell
python -m reformer_tts.cli --help
```

All commands are executed using cli, for example:
```shell
python -m reformer_tts.cli train-vocoder
```

Most parameters (in particular, all training hyperparameters) are specified via
`--config` argument to `cli` (that goes before the command you want to run), eg:
```shell
python -m reformer_tts.cli -c /path/to/your/config.yml train-vocoder
```

Default values can be found in `reformer_tts.config.Config` (and its fields).


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

4. For training, ensure you have CUDA and GPU drivers installed (for details, see instructions on PyTorch website)


### 2. Configure tools

1. In order for dvc to have write access to the remote, configure your gcp account (using credentials from the generated json file):
```shell
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-credentials.json
```

*NOTE: if you only need read acces (for reproduction), you don't need to perform step 1*

2. Get all of the data - this step needs to be repeated:
    - every time you start working after a break
    - after every git pull
    - after checking out another git branch
```shell
dvc pull
```


### 3. Check if the setup is correct

To do this you can run project tests:
```shell script
python -m pytest --pyargs reformer_tts
```

All tests should work on CPU and GPU, and may take up to a minute to complete.

**Remember to pass `--pyargs reformer_tts` to pytest, otherwise it will search data directories for tests**


### Setup details

- Use whatever package manager you want
- Use `Python>=3.8`
- All python dependencies will be in `requirements.txt`
  as well as in `environment.yml`
- One central entrypoint for running tasks: `reformer_tts/cli.py`,
  run `python reformer_tts/cli.py --help` for detailed reference


### Configuration

Configuration is organized in dataclass structures:
- Each project submodule has its own configuration file, called `config.py`,
  where the parameters and *default* values are defined - for example,
  dataset config parameters are specified in `reformer_tts.dataset.config`
- The `reformer_tts.config.Config` class contains all submodules' config settings
- *Actual* values of config parameters are loaded from configuration files in yaml format,
  best practice is to only override defaults in the yaml files

This way, the default values are set close to the place where they are used,
any config value can be overridden wherever you want

**To change runtime configuration**
- automatically generate configuration with default values using command
  `python reformer_tts/cli.py save-config -o config/custom.yml`
  or manually copy one of the existing configuration files in `config/` directory
- remove defaults you don't wish to change from the generated config file
- change values you wish to change in the generated config file
- specify your config when running cli scripts using `-c` option, ie:
  `python reformer_tts/cli.py -c config/custom.yml [COMMAND]`

**To add configuration for new module**
- create `config.py` in your module
- define a dataclass with all necessary config parameters in the new file:
    - make sure your class does not re-define parameter values for other config files
      (ie. we specified number of spectrogram channels only once - in the same place
      for both `dataset` and `squeezewave` modules)
    - make sure your class has default values for all the parameters
- add field for your dataclass in the `reformer_tts.config` main config class


### Data dependencies

We use [DVC](https://dvc.org/) for defining data processing pipelines.
Remote is set up on Google Cloud Storage, for details run `dvc config list`.


### Setup for running jobs on [entropy cluster](entropy.mimuw.edu.pl)

Nodes prepared for running:
* asusgpu3
* asusgpu4
* asusgpu1
* arnold
* sylvester

#### Running trainig on node with homedir
* Clone repo to your homedir
* Make sure dataset path is configured in `/scidatalg`
* Setup command to call file from your homedir
* Commit your changes
* Run sbatch script

#### Running training on specific node without homedir
Before runing:
* chose node from already prepared or prepare new one using instructions below
* copy repository to your home dir
* make sure NEPTUNE API TOKEN is set in your environment

To run training:
* prepare training config and push it onto remote repository
* login to chosen node using interactive session `srun --qos=gsn --partition=common --nodelist=<name_of_chosen_node> --pty /bin/bash`
* goto `/scidatalg/reformer-tts/reformer-tts/` make sure repository is pulled and on proper branch
* log back to login node
* copy and modify `jobs/train_entropy.sbatch` - fill node name and training command
* run `sbatch your/job/script/location.sbatch`

**Pro Tip** `watch -n 1 squeue -u your_username` to watch if your job is already running
**Pro Tip2** You can watch the updates to the log by running `tail -f file.log` or `less --follow-name +F file.log`

#### Pull from dvc
To pull from dvc use `jobs/entropy_dvc_pull.sbatch`.
* copy this file
* fill node name
* adjust dvc command
* run job using sbatch

#### New node preparation

Since /scidatasm directory is not syncing while we want to train we have to setup training on each node separately by hand. To setup env on new node follow this instuctions:

**Note**: only nodes with /scidatalg are supported by this scripts. These nodes are: asusgpu4, asusgpu3, asusgpu2, asusgpu1, arnold, sylvester
* login to node using interactive session `srun --qos=gsn --partition=common --nodelist=<name_of_chosen_node> --pty /bin/bash`
* copy google api credentials to `${HOME}/gcp-cred.json` (using your favourite editor)
* copy the content of `scripts/setup_entropy_node.sh` to new file in home dir (again using editor)
* run copied script
