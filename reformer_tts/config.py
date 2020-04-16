import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path, PosixPath, WindowsPath
from typing import Union

import dacite as D
import yaml

from reformer_tts.dataset.config import DatasetConfig
from reformer_tts.squeeze_wave.config import SqueezeWaveConfig


@dataclass
class Config:
    """
    General class for storing all project configuration.
    Dataclasses (with reasonable defaults) should be implemented in
    submodules (eg. dataset-related config in reformer_tts.dataset).
    """

    data_directory: Path = Path("data")
    """ Root folder of the dataset """

    raw_data_directory: Path = data_directory / "raw"
    """ Data for pipeline stage: after download """
    video_directory: Path = raw_data_directory / "videos"
    transcript_directory: Path = raw_data_directory / "transcripts"

    preprocessed_data_directory: Path = data_directory / "preprocessed"
    """ Data for pipeline stage: after preprocessing """
    merged_transcript_csv_path: Path = preprocessed_data_directory / "transcript.csv"
    audio_directory: Path = preprocessed_data_directory / "audio"
    mel_directory: Path = preprocessed_data_directory / "mel"

    nltk_data_directory: Path = Path(".nltk")
    """ Directory where NLTK will store downloaded data for text processing """

    dataset: DatasetConfig = DatasetConfig()
    # TODO: Add config for reformer model here
    squeeze_wave: SqueezeWaveConfig = SqueezeWaveConfig()

    def to_yaml_fle(self, path: Union[str, Path]):
        """  Saves current config (incl. defaults) to yaml file at path """
        config_dict = asdict(self)
        with open(path, "w") as config_file:
            # safe_dump will not work with python types: Path and Tuple
            yaml.dump(config_dict, config_file)

    @classmethod
    def from_yaml_file(cls, path: Union[str, Path]):
        """
        Loads config from yaml file, deep-casting it into correct dataclasses.
        Validates that all keys from yaml are present in target dataclass.
        """
        with open(path, "r") as config_file:
            # safe_load will not work with python types: Path and Tuple
            # unsafe_load does not work: https://github.com/yaml/pyyaml/issues/266
            config_dict = yaml.load(config_file, Loader=yaml.Loader)

        config = D.from_dict(
            data_class=cls,
            data=config_dict,
            config=D.Config(check_types=False, strict=True)
        )
        return config

    @classmethod
    def from_python_module(
            cls, path: str, config_variable_name: str = "CONFIG"
    ) -> "Config":
        """
        Loads configuration from a variable `config_variable_name`,
        from the python file at `path`.
        """
        # import module (Python >= 3.5)
        spec = importlib.util.spec_from_file_location("module.name", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # from {path} import {config_variable_name}
        config = getattr(config_module, config_variable_name)
        return config


def _path_representer(dumper: yaml.Dumper, path: Path):
    return dumper.represent_scalar(u"!path", str(path))


def _path_constructor(loader: yaml.Loader, node):
    value = loader.construct_scalar(node)
    return Path(value)


yaml.add_representer(Path, _path_representer)
yaml.add_representer(PosixPath, _path_representer)
yaml.add_representer(WindowsPath, _path_representer)
yaml.add_constructor(u"!path", _path_constructor)
