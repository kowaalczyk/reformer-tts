from pathlib import Path

import click
import yaml
from click import Context, ClickException

from reformer_tts.scrapper.download import download_speech_videos_and_transcripts

DATA_DIRECTORY = Path("data")
CONFIG_PATH = DATA_DIRECTORY / "config.yaml"


@click.group()
@click.pass_context
def cli(ctx: Context):
    ctx.ensure_object(dict)
    ctx.obj["CONFIG"] = load_config(CONFIG_PATH)


@cli.command()
@click.pass_context
def download(ctx: Context):
    config = ctx.obj["CONFIG"]
    url = config["transcripts_list_url"]
    download_speech_videos_and_transcripts(url)


def load_config(config_path: Path):
    with config_path.open("r") as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as e:
            raise ClickException(str(e))


if __name__ == "__main__":
    cli(obj={})
