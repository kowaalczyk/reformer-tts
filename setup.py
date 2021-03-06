from setuptools import setup, find_packages

setup(
    name="reformer_tts",
    version="0.1",
    packages=find_packages(include=('reformer_tts', 'reformer_tts.*')),
    python_requires=">=3.8",
    install_requires=[
        "dacite==1.4.0",
        "dvc==0.88",
        "Click==7",
        "pytorch-lightning==0.7.6",
        "PyYAML==5.1.2",
        "tqdm==4.43.0",
        "beautifulsoup4==4.8.2",
        "requests==2.23.0",
        "reformer-pytorch==0.19.1",
        "demjson==2.2.4",
        "torch==1.4.0",
        "torchvision==0.5.0",
        "torchaudio==0.4.0",
        "scipy==1.4.1",
        "ffmpeg-python==0.2.0",
        "matplotlib==3.1.3",
        "librosa==0.7.2",
        "unidecode==1.1.1",
        "nltk==3.4.5",
        "g2p-en==2.1.0",
        "pydub==0.23.1",
        "psutil==5.7.0",
        "pandas==1.0.3",
        "google-cloud-storage==1.28.1",
        "pytest==5.4.2",
        "transformers==2.11.0",
    ],
    entry_points="""
        [console_scripts]
        reformercli=reformer_tts.cli:cli
    """,
)
