from dataclasses import asdict
from pprint import pprint

import click
import torch
from click import Context
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

from reformer_tts.config import Config
from reformer_tts.dataset.download import download_speech_videos_and_transcripts
from reformer_tts.dataset.preprocess import preprocess_data
from reformer_tts.training.wrappers import LitSqueezeWave


@click.group()
@click.option("-c", "--config", envvar="REFORMER_TTS_CONFIG", default="config/default.yml")
@click.pass_context
def cli(ctx: Context, config):
    ctx.ensure_object(dict)
    ctx.obj["CONFIG"] = Config.from_yaml_file(config)


@cli.command()
@click.pass_context
def download(ctx: Context):
    config = ctx.obj["CONFIG"]
    download_speech_videos_and_transcripts(
        url=config.dataset.source_url,
        transcript_directory=config.dataset.structure.transcript_directory,
        video_directory=config.dataset.structure.video_directory
    )


@cli.command()
@click.pass_context
def preprocess(ctx: Context):
    config = ctx.obj["CONFIG"]
    preprocess_data(
        trump_speaker_names=config.dataset.trump_speaker_names,
        transcript_directory=config.transcript_directory,
        merged_transcript_csv_path=config.merged_transcript_csv_path,
        audio_directory=config.audio_directory,
        video_directory=config.video_directory,
        spectrogram_dir=config.mel_directory,
        nltk_data_directory=config.nltk_data_directory,
        audio_format=config.dataset.audio_format,
        mel_format=config.dataset.mel_format,
        use_tacotron2_spectrograms=config.dataset.use_tacotron2_spectrograms
    )


@cli.command()
@click.option("-e", "--experiment", "experiment_name", default="squeeze_wave_training")
@click.pass_context
def train_vocoder(ctx: Context, experiment_name: str):
    config = ctx.obj["CONFIG"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        on_gpu = True
        gpus = 1  # todo: config?
    else:
        on_gpu = False
        gpus = 0

    model = LitSqueezeWave(config, on_gpu=on_gpu)
    logger = NeptuneLogger(
        project_name="reformer-tts/reformer-tts",
        experiment_name=experiment_name,
        params={
            **asdict(config),
            **asdict(config.dataset),
            **asdict(config.squeeze_wave),
            **asdict(config.vocoder_training),
        }
    )
    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=10,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
    )
    trainer = Trainer(
        gpus=gpus,
        logger=logger,
        log_save_interval=50,
        row_log_interval=5,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model)


@cli.command()
@click.pass_context
def show_config(ctx: Context):
    config = ctx.obj["CONFIG"]
    pprint(asdict(config))


@cli.command()
@click.option("-o", "--output", type=str, required=True, help="Path where config will be saved")
@click.pass_context
def save_config(ctx: Context, output):
    """ Save all config variables (defaults + overrides from config file) """
    config = ctx.obj["CONFIG"]
    config.to_yaml_fle(output)
    print(f"Config saved to {output}")


if __name__ == "__main__":
    cli(obj={})
