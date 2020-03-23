from pathlib import Path

import ffmpeg
from tqdm.auto import tqdm


def video_to_audio(videos_dir: Path, audio_dir: Path):
    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)

    video_paths = list(videos_dir.glob("*.mp4"))
    for video_path in tqdm(video_paths):
        input_filename = str(video_path)
        output_filename = str(
            Path(audio_dir / video_path.name).with_suffix(".wav")
        )
        ffmpeg.input(input_filename).audio.output(output_filename)\
            .run(quiet=True, overwrite_output=True)
