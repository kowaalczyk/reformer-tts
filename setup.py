from setuptools import setup, find_packages

setup(
    name="reformer_tts",
    version="0.1",
    packages=find_packages(include=('reformer_tts', 'reformer_tts.*')),
    python_requires=">=3.8",
    install_requires=[],
    entry_points="""
        [console_scripts]
    """,
)
