import csv
import shutil
import tarfile
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import click
from tqdm.auto import tqdm

import reformer_tts.dataset.convert as C
from reformer_tts.config import Config
from reformer_tts.dataset.config import AudioFormat, MelFormat


def preprocess_lj_speech_data(
        archive_path: Path,
        transcript_csv_path: Path,
        audio_directory: Path,
        mel_directory: Path,
        nltk_data_directory: Path,
        audio_format: AudioFormat,
        mel_format: MelFormat,
):
    phonemizer = C.PhonemeSequenceCreator(nltk_data_directory)
    mel_creator = C.MelSpectrogramCreator(audio_format.sampling_rate, **asdict(mel_format))

    with TemporaryDirectory() as temporary_directory:
        with tarfile.open(archive_path, "r:bz2") as archive:
            for archived_file in tqdm(archive, desc=f"Unzipping {archive_path.name}", unit="file"):
                archive.extract(archived_file, path=temporary_directory)

        temporary_audio_directory = Path(temporary_directory) / "LJSpeech-1.1" / "wavs"
        shutil.move(str(temporary_audio_directory), str(audio_directory))

        temporary_transcript_csv_path = Path(temporary_directory) / "LJSpeech-1.1" / "metadata.csv"
        with temporary_transcript_csv_path.open("r") as input_file:
            reader = csv.reader(input_file, delimiter="|", quoting=csv.QUOTE_NONE)
            input_rows = list(reader)

    with transcript_csv_path.open("w") as output_file:
        writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["title", "audio_path", "text", "phonemes"])
        for row in tqdm(input_rows, desc="Preprocessing transcript", unit="row"):
            title, _, text = row
            audio_path = str(audio_directory / f"{title}.wav")
            phonemes = " ".join(phonemizer.phonemize(text))
            writer.writerow([title, audio_path, text, phonemes])

    mel_directory.mkdir(parents=True, exist_ok=True)
    audio_files = list(audio_directory.iterdir())
    for audio_file in tqdm(audio_files, desc="Generating mel spectrograms", unit="clip"):
        try:
            spectrogram_file = (mel_directory / audio_file.name).with_suffix(".pt")
            mel_creator.audio_to_mel_spectrogram(audio_file, spectrogram_file)
        except Exception as e:
            print(f"{e}, {audio_file =}")
            return


@click.command()
@click.option("-c", "--config", "config_path", default="config/lj_speech.yml",)
@click.option("-i", "--input", "archive_path", default=None)
def main(config_path: str, archive_path: Optional[str]):
    config = Config.from_yaml_file(config_path)
    if archive_path is None:
        archive_path = config.data_directory / "LJSpeech-1.1.tar.bz2"
    else:
        archive_path = Path(archive_path)
    preprocess_lj_speech_data(
        archive_path=archive_path,
        transcript_csv_path=config.merged_transcript_csv_path,
        audio_directory=config.audio_directory,
        mel_directory=config.mel_directory,
        nltk_data_directory=config.nltk_data_directory,
        audio_format=config.dataset.audio_format,
        mel_format=config.dataset.mel_format
    )

    print(f"Saved transcript csv to: {config.merged_transcript_csv_path.resolve()}")
    print(f"Saved audio clips to {config.audio_directory.resolve()}")
    print(f"Saved mel spectrograms to {config.mel_directory.resolve()}")


if __name__ == "__main__":
    main()
