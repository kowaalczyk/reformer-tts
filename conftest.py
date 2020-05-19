import pytest
from reformer_tts.config import Config


@pytest.fixture(scope="session", params=["config/trump-tacotron2.yml", "config/trump-tacotron2.yml"])
def config(request):
    return Config.from_yaml_file(request.param)
