import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple

import ffmpeg
import nltk
from nltk import tokenize
from pydub import AudioSegment
from tqdm import tqdm

from reformer_tts.scrapper.download import TRANSCRIPT_DIRECTORY, VIDEO_DIRECTORY

DATA_DIRECTORY = Path("data")
PREPROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "preprocessed"
AUDIO_DIRECTORY = PREPROCESSED_DATA_DIRECTORY / "audio"
MERGED_TRANSCRIPT_CSV_PATH = PREPROCESSED_DATA_DIRECTORY / "transcript.csv"
NLTK_DATA_DIRECTORY = Path(".nltk")

TRUMP_SPEAKER_NAMES = ["Donald Trump", "President Trump"]


def setup_g2p_phonemizer(nltk_data_directory: Path):
    # workaround for https://github.com/Kyubyong/g2p/issues/12
    nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_directory)
    nltk.download("cmudict", download_dir=nltk_data_directory)
    nltk.download("punkt", download_dir=nltk_data_directory)
    nltk.data.path.append(nltk_data_directory.resolve())
    from g2p_en import G2p
    return G2p()


g2p = setup_g2p_phonemizer(NLTK_DATA_DIRECTORY)


def preprocess_data():
    transcript_paths = list(TRANSCRIPT_DIRECTORY.glob("*.json"))
    transcripts = [load_transcript(transcript_path) for transcript_path in transcript_paths]

    processed_transcripts = [
        preprocess_transcript(transcript)
        for transcript in tqdm(transcripts, desc="Processing transcripts", unit="transcript")
    ]

    PREPROCESSED_DATA_DIRECTORY.mkdir(exist_ok=True, parents=True)
    speech_names = [transcript_path.stem for transcript_path in transcript_paths]
    save_merged_transcript_csv(
        processed_transcripts,
        speech_names,
        output_path=MERGED_TRANSCRIPT_CSV_PATH
    )

    with TemporaryDirectory() as temporary_directory:
        speech_audio_directory = Path(temporary_directory)
        for speech_name in tqdm(speech_names, desc="Converting video to audio", unit="video"):
            convert_mp4_to_wav(
                input_path=VIDEO_DIRECTORY / f"{speech_name}.mp4",
                output_path=speech_audio_directory / f"{speech_name}.wav"
            )

        AUDIO_DIRECTORY.mkdir(exist_ok=True, parents=True)
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
                    output_path=AUDIO_DIRECTORY / f"{speech_name}_{i:04d}.wav",
                    start=row["start"],
                    end=row["end"],
                )

    print(f"Saved speech audio clips to {AUDIO_DIRECTORY.resolve()}")
    print(f"Saved merged transcript csv to: {MERGED_TRANSCRIPT_CSV_PATH.resolve()}")


def load_transcript(transcript_path: Path) -> Dict:
    with transcript_path.open("r") as transcript_file:
        return json.load(transcript_file)


def preprocess_transcript(transcript: Dict) -> List[Dict]:
    speaker_mapping = transcript["speakerMap"]
    rows = []
    for node in transcript["nodes"]:
        if node["type"] == "Monologue":
            speaker_name = get_node_speaker_name(node, speaker_mapping)
            if speaker_name in TRUMP_SPEAKER_NAMES:
                node_rows = preprocess_transcript_node(node)
                rows += node_rows
    return rows


def get_node_speaker_name(node: Dict, speaker_mapping: Dict) -> str:
    speaker_data, _, _ = node["nodes"]
    speaker_id = str(speaker_data["data"]["speakerId"])
    speaker_name = speaker_mapping[speaker_id]["name"]
    return speaker_name


def preprocess_transcript_node(node: Dict) -> List[Dict]:
    sentence_texts, sentence_timestamps, sentence_phonemes = parse_transcript_node(node)
    rows = []
    for text, timestamp, phonemes in zip(sentence_texts, sentence_timestamps, sentence_phonemes):
        row = get_sentence_row(text, timestamp, phonemes)
        rows.append(row)
    return rows


def parse_transcript_node(node: Dict) -> Tuple[List, List, List]:
    _, _, text_data = node["nodes"]
    timestamps = text_data["data"]["Timestamps"]
    text = text_data["nodes"][0]["leaves"][0]["text"]
    sentence_texts = tokenize.sent_tokenize(text)
    sentence_timestamps = get_sentence_timestamps(timestamps, sentence_texts)
    sentence_phonemes = [phonemize(text) for text in sentence_texts]
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


def write_transcript_rows(writer, transcript: List[Dict], speech_name: str):
    for i, row in enumerate(transcript):
        audio_clip_path = AUDIO_DIRECTORY / f"{speech_name}_{i:04d}.wav"
        values = [row[key] for key in ["text", "start", "end", "phonemes"]]
        writer.writerow([speech_name, audio_clip_path, *values])


def save_merged_transcript_csv(
        processed_transcripts: List[List[Dict]],
        speech_names: List[str],
        output_path: Path
):
    with output_path.open("w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["speech", "audio_path", "text", "start", "end", "phonemes"])
        for transcript, speech_name in zip(processed_transcripts, speech_names):
            write_transcript_rows(writer, transcript, speech_name)


def save_audio_clip(input_path: Path, output_path: Path, start: int, end: int):
    input_audio = AudioSegment.from_wav(input_path)
    audio_clip = input_audio[start:end]
    audio_clip.export(output_path, format="wav")


def convert_mp4_to_wav(input_path: Path, output_path: Path):
    audio = ffmpeg.input(str(input_path)).audio
    audio.output(str(output_path)).run(quiet=True, overwrite_output=True)


def phonemize(text: str) -> List[str]:
    return g2p(text)
