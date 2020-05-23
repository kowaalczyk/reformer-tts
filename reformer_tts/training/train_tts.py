import os

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging.neptune import NeptuneLogger

from reformer_tts.training.wrappers import LitReformerTTS
from reformer_tts.config import Config


def train_tts(config: Config, checkpoint_to_resume: Optional[Path]):

    seed_everything(42)

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        on_gpu = True
        gpus = 1  # todo: config?
    else:
        on_gpu = False
        gpus = 0

    max_epochs = config.experiment.max_epochs
    neptune_logger = NeptuneLogger(
        project_name="reformer-tts/reformer-tts",
        experiment_name=config.experiment.experiment_name,
        params={
            **asdict(config),
            **asdict(config.dataset),
            **asdict(config.model),
            **asdict(config.experiment.tts_training),
        },
        tags=["reformer-tts"] + config.experiment.tags.split()
    )

    if checkpoint_to_resume is None:
        model = LitReformerTTS(config, on_gpu=on_gpu)
    else:
        model = LitReformerTTS.load_from_checkpoint(
            str(checkpoint_to_resume),
            config=config,
            on_gpu=on_gpu
        )
    experiment_dir = os.path.join(config.experiment.checkpoints_dir, config.experiment.experiment_name)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(experiment_dir, "{epoch}-{val_loss:.2f}"),
        save_top_k=config.experiment.save_top_k_checkpoints,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='',
    )

    if config.experiment.tts_training.early_stopping_epochs is not None:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=config.experiment.tts_training.early_stopping_epochs,
            verbose=True,
        )
    else:
        early_stop_callback = False

    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        logger=neptune_logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        log_save_interval=50,
        row_log_interval=5,
    )
    trainer.fit(model)
