from dataclasses import asdict
from pprint import pprint

import click
from click import Context

from reformer_tts.config import Config
from reformer_tts.dataset.download import download_speech_videos_and_transcripts
from reformer_tts.dataset.preprocess import preprocess_data


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
        mel_directory=config.mel_directory,
        nltk_data_directory=config.nltk_data_directory,
        audio_format=config.dataset.audio_format,
        mel_format=config.dataset.mel_format
    )


@cli.command()
@click.pass_context
def show_config(ctx: Context):
    config = ctx.obj["CONFIG"]
    pprint(asdict(config))


@cli.command()
@click.option("-o", "--output", type=str, required=True,
              help="Path where config will be saved")
@click.pass_context
def save_config(ctx: Context, output):
    """ Save all config variables (defaults + overrides from config file) """
    config = ctx.obj["CONFIG"]
    config.to_yaml_fle(output)
    print(f"Config saved to {output}")


if __name__ == "__main__":
    cli(obj={})
