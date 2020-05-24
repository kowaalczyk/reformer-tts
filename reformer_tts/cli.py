import sys
from dataclasses import asdict
from pathlib import Path
from pprint import pprint
from typing import Optional

import click
import matplotlib.pyplot as plt
import torch
import torchaudio
from click import Context
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from torch.nn.functional import mse_loss
from tqdm import trange, tqdm

from reformer_tts.config import Config
from reformer_tts.dataset.convert import PhonemeSequenceCreator
from reformer_tts.dataset.download import download_speech_videos_and_transcripts
from reformer_tts.dataset.preprocess import preprocess_data
from reformer_tts.dataset.visualize import plot_spectrogram
from reformer_tts.squeeze_wave.modules import SqueezeWave
from reformer_tts.training.train_tts import train_tts as train_tts_function
from reformer_tts.training.wrappers import LitSqueezeWave, LitReformerTTS


@click.group()
@click.option("-c", "--config", envvar="REFORMER_TTS_CONFIG", default="config/default.yml")
@click.pass_context
def cli(ctx: Context, config):
    ctx.ensure_object(dict)
    ctx.obj["CONFIG"] = Config.from_yaml_file(config)


@cli.command()
@click.option("-r", "--resume", type=str, default=None, help="Path to checkpoint to resume")
@click.pass_context
def train_tts(ctx: Context, resume: Optional[str]):
    config = ctx.obj["CONFIG"]
    if resume is not None:
        resume = Path(resume)
    train_tts_function(config, resume)


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
    config: Config = ctx.obj["CONFIG"]

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
            **asdict(config.experiment.vocoder_training),
        }
    )
    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=config.experiment.n_saved_models,
        verbose=True,
    )
    if config.experiment.early_stopping_epochs is not None:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=config.experiment.early_stopping_epochs,
            verbose=True,
        )
    else:
        early_stop_callback = False

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
@click.option("-r", "--reformer-checkpoint", type=str, required=True, help="Path to reformer checkpoint")
@click.option("-s", "--squeeze-wave-checkpoint", type=str, required=True, help="Path to squeezewave checkpoint")
@click.option("-o", "--output-dir", type=str, required=True, help="Path where outputs will be saved")
@click.option("-m", "--max-samples", type=int, default=None, help="Maximum number of total generated samples")
@click.pass_context
def predict_samples(
        ctx: Context,
        reformer_checkpoint: str,
        squeeze_wave_checkpoint: str,
        output_dir: str,
        max_samples: Optional[int]
):
    """
    Generates predictions on the test_set portion of text-to-spectrogram dataset.

    Provided config must be compatible with both reformer and squeezewave (keys and
    values in config structure must be the same as the ones used during their training)
    """
    config = ctx.obj["CONFIG"]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device('cuda')
        on_gpu = True
    else:
        device = torch.device('cpu')
        on_gpu = False

    reformer = LitReformerTTS.load_from_checkpoint(reformer_checkpoint, config=config)
    reformer = reformer.eval()
    squeeze_wave = LitSqueezeWave.load_from_checkpoint(
        squeeze_wave_checkpoint,
        config=config,
        on_gpu=on_gpu
    )
    squeeze_wave = SqueezeWave.remove_norms(squeeze_wave.model)
    squeeze_wave = squeeze_wave.eval()

    results = list()
    reformer.prepare_data()

    if len(reformer.test_set) == 0:
        dataset = reformer.val_set
    else:
        dataset = reformer.test_set

    if max_samples is None:
        max_samples = len(dataset)

    with torch.no_grad():
        # todo: use ReformerTTS.infer
        for test_sample_idx in trange(max_samples, desc="predicting"):
            sample = dataset[test_sample_idx]

            phonemes_in = sample['phonemes'].unsqueeze(0).to(device=device)
            spectrogram_in = sample['spectrogram'].unsqueeze(0).to(device=device)

            # todo: we shouldn't pass target spectrogram into reformer:
            spectrogram_out, stop_out = reformer(phonemes_in, spectrogram_in[:, :-1, :])
            mse = mse_loss(spectrogram_out, spectrogram_in[:, 1:, :])

            cutoff: int = stop_out.argmax()
            spectrogram_out: torch.Tensor = spectrogram_out.transpose(1, 2)
            spectrogram_out = spectrogram_out[:, :, :cutoff]

            audio_out = squeeze_wave.infer(spectrogram_out)
            results.append({
                "spectrogram": spectrogram_out.cpu(),
                "spectrogram_mse": float(mse.cpu().numpy()),
                "audio": audio_out.cpu(),
                "idx": sample["idx"],
            })
        best_mse = min(results, key=lambda r: r["spectrogram_mse"])["spectrogram_mse"]
        worst_mse = max(results, key=lambda r: r["spectrogram_mse"])["spectrogram_mse"]
        mean_mse = sum(r["spectrogram_mse"] for r in results) / float(len(results))
        print(f"{best_mse=:.4f}, {worst_mse=:.4f}, {mean_mse=:.4f}")

        for result in tqdm(results, desc="saving"):
            filename = f"pred-{result['idx']}-idx_{result['spectrogram_mse']:.4f}-mse"

            spectrogram_path = output_dir / f"{filename}.png"
            plot_spectrogram(result["spectrogram"], scale=False)
            plt.savefig(str(spectrogram_path))
            plt.close()

            audio_path = output_dir / f"{filename}.wav"
            torchaudio.save(
                str(audio_path),
                result["audio"],
                config.dataset.audio_format.sampling_rate
            )
        print(f"Results saved to {output_dir.resolve()}")


@cli.command()
@click.option("-r", "--reformer-checkpoint", type=str, required=True, help="Path to reformer checkpoint")
@click.option("-s", "--squeeze-wave-checkpoint", type=str, required=True, help="Path to squeezewave checkpoint")
@click.option("-o", "--output-dir", type=str, required=True, help="Path where outputs will be saved")
@click.pass_context
def predict_from_text(
        ctx: Context,
        reformer_checkpoint: str,
        squeeze_wave_checkpoint: str,
        output_dir: str,
):
    config: Config = ctx.obj["CONFIG"]

    # todo: refactor - most of this is the same as in predict_samples
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        on_gpu = True
        device = torch.device('cuda')
    else:
        on_gpu = False
        device = torch.device('cpu')

    reformer = LitReformerTTS.load_from_checkpoint(reformer_checkpoint, config=config)
    reformer = reformer.eval()

    squeeze_wave = LitSqueezeWave.load_from_checkpoint(
        squeeze_wave_checkpoint,
        config=config,
        on_gpu=on_gpu
    )
    squeeze_wave = SqueezeWave.remove_norms(squeeze_wave.model)
    squeeze_wave = squeeze_wave.eval()

    phonemizer = PhonemeSequenceCreator(config.nltk_data_directory)
    phoneme_encoder = reformer.get_phoneme_encoder()

    print("Type a sentence and press enter to convert it to speech:")
    with torch.no_grad():
        for idx, line in enumerate(sys.stdin):
            phonemes = phonemizer.phonemize(line)
            print(f"Predicting from {phonemes=}...")
            phonemes = " ".join(phonemes)
            phonemes = phoneme_encoder(phonemes).unsqueeze(0).to(device=device)

            spectrogram, stop = reformer.model.infer(
                phonemes, combine_strategy="concat", verbose=True
            )
            spectrogram = spectrogram[:, :, :stop.item()]
            audio_out = squeeze_wave.infer(spectrogram)

            spectrogram_path = output_dir / f"pred-stdin-{idx}.png"
            plot_spectrogram(spectrogram.cpu(), scale=False)
            plt.savefig(str(spectrogram_path))
            plt.close()

            audio_path = output_dir / f"pred-stdin-{idx}.wav"
            torchaudio.save(
                str(audio_path),
                audio_out.cpu(),
                config.dataset.audio_format.sampling_rate
            )
            print(f"Output saved to {audio_path.resolve()}")


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
    config.to_yaml_file(output)
    print(f"Config saved to {output}")


if __name__ == "__main__":
    cli(obj={})
