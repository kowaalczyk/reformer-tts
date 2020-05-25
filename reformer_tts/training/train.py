from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.logging.neptune import NeptuneLogger

from reformer_tts.config import Config
from reformer_tts.training.wrappers import LitReformerTTS, LitSqueezeWave


def train_tts(config: Config, checkpoint_to_resume: Optional[Path]):
    seed_everything(42)
    on_gpu, gpus = setup_cuda()

    if checkpoint_to_resume is None:
        model = LitReformerTTS(config, on_gpu=on_gpu)
    else:
        model = LitReformerTTS.load_from_checkpoint(
            str(checkpoint_to_resume),
            config=config,
            on_gpu=on_gpu
        )
    logger = setup_logger(config, additional_tags=["reformer-tts"])
    trainer = setup_trainer(config, gpus, logger)
    trainer.fit(model)


def train_vocoder(config: Config, checkpoint_path: Optional[Path]):
    seed_everything(42)
    on_gpu, gpus = setup_cuda()

    if checkpoint_path is None:
        model = LitSqueezeWave(config, on_gpu=on_gpu)
    else:
        model = LitSqueezeWave.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            config=config,
            on_gpu=on_gpu,
        )
    logger = setup_logger(config, additional_tags=["reformer-tts"])
    trainer = setup_trainer(config, gpus, logger)
    trainer.fit(model)


def setup_cuda() -> Tuple[bool, int]:
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        on_gpu = True
        gpus = 1  # todo: config?
    else:
        on_gpu = False
        gpus = 0
    return on_gpu, gpus


def setup_logger(config: Config, additional_tags: Optional[List[str]] = None):
    if additional_tags is None:
        additional_tags = []
    tags = additional_tags + config.experiment.tags.split()
    return NeptuneLogger(
        project_name="reformer-tts/reformer-tts",
        experiment_name=config.experiment.experiment_name,
        params={
            **asdict(config),
            **asdict(config.dataset),
            **asdict(config.model),
            **asdict(config.experiment.tts_training),
        },
        tags=tags
    )


def setup_trainer(config: Config, gpus: int, logger: LightningLoggerBase) -> Trainer:
    checkpoint_callback, early_stop_callback = setup_callbacks(config)
    return Trainer(
        gpus=gpus,
        max_epochs=config.experiment.max_epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        log_save_interval=50,
        row_log_interval=5,
    )


def setup_callbacks(config: Config) -> Tuple[ModelCheckpoint, EarlyStopping]:
    experiment_dir = Path(config.experiment.checkpoints_dir) / config.experiment.experiment_name
    filename_format = str(experiment_dir / "{epoch}-{val_loss:.2f}")
    checkpoint_callback = ModelCheckpoint(
        filepath=filename_format,
        save_top_k=config.experiment.save_top_k_checkpoints,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    if config.experiment.tts_training.early_stopping_epochs is not None:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=config.experiment.tts_training.early_stopping_epochs,
            verbose=True,
        )
    else:
        early_stop_callback = False

    return checkpoint_callback, early_stop_callback
