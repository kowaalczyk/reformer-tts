import csv
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple, Container

from nltk import tokenize
from pydub import AudioSegment
from tqdm import tqdm

import reformer_tts.dataset.convert as C
from reformer_tts.dataset.config import AudioFormat, MelFormat


def preprocess_data(
        trump_speaker_names: Container[str],
        transcript_directory: Path,
        merged_transcript_csv_path: Path,
        audio_directory: Path,
        video_directory: Path,
        spectrogram_dir: Path,
        nltk_data_directory: Path,
        audio_format: AudioFormat,
        mel_format: MelFormat,
        use_tacotron2_spectrograms: bool
):
    transcript_paths = list(transcript_directory.glob("*.json"))
    transcripts = [load_transcript(transcript_path) for transcript_path in transcript_paths]

    phonemizer = C.PhonemeSequenceCreator(nltk_data_directory)
    processed_transcripts = [
        preprocess_transcript(
            transcript,
            phonemizer,
            trump_speaker_names,
            audio_format.min_duration_ms,
            audio_format.max_duration_ms
        )
        for transcript in tqdm(transcripts, desc="Processing transcripts", unit="transcript")
    ]

    merged_transcript_csv_path.parent.mkdir(exist_ok=True, parents=True)
    speech_names = [transcript_path.stem for transcript_path in transcript_paths]
    save_merged_transcript_csv(
        processed_transcripts,
        speech_names,
        output_path=merged_transcript_csv_path,
        audio_directory=audio_directory
    )

    with TemporaryDirectory() as temporary_directory:
        speech_audio_directory = Path(temporary_directory)
        for speech_name in tqdm(speech_names, desc="Converting video to audio", unit="video"):
            C.mp4_to_wav(
                input_path=video_directory / f"{speech_name}.mp4",
                output_path=speech_audio_directory / f"{speech_name}.wav"
            )

        audio_directory.mkdir(exist_ok=True, parents=True)
        for transcript, speech_name in tqdm(
                zip(processed_transcripts, speech_names),
                desc="Splitting speech audio into clips",
                unit="speech",
                total=len(processed_transcripts),
        ):
            for i, row in tqdm(
                    enumerate(transcript),
                    desc="Saving clips",
                    unit="clip",
                    total=len(transcript),
                    leave=False,
            ):
                save_audio_clip(
                    input_path=speech_audio_directory / f"{speech_name}.wav",
                    output_path=audio_directory / f"{speech_name}_{i:04d}.wav",
                    start=row["start"],
                    end=row["end"],
                )

    audio_files = list(audio_directory.glob("*.wav"))
    for audio_file in tqdm(audio_files, desc="Resampling", unit="clip"):
        try:
            C.resample_wav(audio_file, audio_file)
        except Exception as e:
            print(f"{e}, {audio_file =}")
            return

    if use_tacotron2_spectrograms:
        spectrogram_factory = C.Tacotron2SpectrogramCreator(
            audio_format.sampling_rate,
            **asdict(mel_format)
        )
    else:
        spectrogram_factory = C.MelSpectrogramCreator(
            audio_format.sampling_rate,
            **asdict(mel_format)
        )
    spectrogram_dir.mkdir(exist_ok=True, parents=True)
    for audio_file in tqdm(
            audio_files,
            desc="Generating mel spectrograms",
            unit="clip"
    ):
        try:
            spectrogram_file = Path(spectrogram_dir / audio_file.name) \
                .with_suffix(".pt")
            spectrogram_factory.audio_to_mel_spectrogram(audio_file, spectrogram_file)
        except Exception as e:
            print(f"{e}, {audio_file =}")
            return

    print(f"Saved merged transcript csv to: {merged_transcript_csv_path.resolve()}")
    print(f"Saved speech audio clips to {audio_directory.resolve()}")
    spectrogram_type = "tacotron2-compatible" if use_tacotron2_spectrograms else "mel"
    print(f"Saved {spectrogram_type} spectrograms to {spectrogram_dir.resolve()}")


def load_transcript(transcript_path: Path) -> Dict:
    with transcript_path.open("r") as transcript_file:
        return json.load(transcript_file)


def preprocess_transcript(
        transcript: Dict,
        phonemizer: C.PhonemeSequenceCreator,
        trump_speaker_names: Container[str],
        min_duration_ms: int,
        max_duration_ms: int
) -> List[Dict]:
    speaker_mapping = transcript["speakerMap"]
    rows = []
    for node in transcript["nodes"]:
        if node["type"] == "Monologue":
            speaker_name = get_node_speaker_name(node, speaker_mapping)
            if speaker_name in trump_speaker_names:
                node_rows = preprocess_transcript_node(
                    node,
                    phonemizer,
                    min_duration_ms,
                    max_duration_ms
                )
                rows += node_rows
    return rows


def get_node_speaker_name(node: Dict, speaker_mapping: Dict) -> str:
    speaker_data, _, _ = node["nodes"]
    speaker_id = str(speaker_data["data"]["speakerId"])
    speaker_name = speaker_mapping[speaker_id]["name"]
    return speaker_name


def preprocess_transcript_node(
        node: Dict,
        phonemizer: C.PhonemeSequenceCreator,
        min_duration_ms: int,
        max_duration_ms: int,
) -> List[Dict]:
    sentence_texts, sentence_timestamps, sentence_phonemes = parse_transcript_node(node, phonemizer)
    rows = []
    for text, timestamp, phonemes in zip(sentence_texts, sentence_timestamps, sentence_phonemes):
        duration = timestamp[-1] - timestamp[0]
        if min_duration_ms <= duration <= max_duration_ms:
            row = get_sentence_row(text, timestamp, phonemes)
            rows.append(row)
    return rows


def parse_transcript_node(node: Dict, phonemizer: C.PhonemeSequenceCreator) -> Tuple[List, List, List]:
    _, _, text_data = node["nodes"]
    timestamps = text_data["data"]["Timestamps"]
    text = text_data["nodes"][0]["leaves"][0]["text"]
    sentence_texts = tokenize.sent_tokenize(text)
    sentence_timestamps = get_sentence_timestamps(timestamps, sentence_texts)
    sentence_phonemes = [phonemizer.phonemize(text) for text in sentence_texts]
    return sentence_texts, sentence_timestamps, sentence_phonemes


def get_sentence_timestamps(timestamps: List[Dict], sentences: List) -> List[Tuple[int, int]]:
    sentence_timestamps = []
    current_index = 0
    for sentence in sentences:
        words = str(sentence).strip().split()
        n_words = len(words)
        start = timestamps[current_index]["Start"]
        end = timestamps[current_index + n_words - 1]["End"]
        sentence_timestamps.append((start, end))
        current_index += n_words
    assert current_index == len(timestamps)
    return sentence_timestamps


def get_sentence_row(text: str, timestamp: Tuple[int, int], phonemes: List) -> Dict:
    start, end = timestamp
    phonemes = " ".join(phonemes)
    return {
        "text": text,
        "start": start,
        "end": end,
        "phonemes": phonemes
    }


def write_transcript_rows(
        writer,
        transcript: List[Dict],
        speech_name: str,
        audio_directory: Path
):
    for i, row in enumerate(transcript):
        audio_clip_path = audio_directory / f"{speech_name}_{i:04d}.wav"
        values = [row[key] for key in ["text", "start", "end", "phonemes"]]
        writer.writerow([speech_name, audio_clip_path, *values])


def save_merged_transcript_csv(
        processed_transcripts: List[List[Dict]],
        speech_names: List[str],
        output_path: Path,
        audio_directory: Path  # for filename formatting
):
    with output_path.open("w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["speech", "audio_path", "text", "start", "end", "phonemes"])
        for transcript, speech_name in zip(processed_transcripts, speech_names):
            write_transcript_rows(writer, transcript, speech_name, audio_directory)


def save_audio_clip(input_path: Path, output_path: Path, start: int, end: int):
    input_audio = AudioSegment.from_wav(input_path)
    audio_clip = input_audio[start:end]
    audio_clip.export(output_path, format="wav")
