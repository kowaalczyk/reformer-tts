import json
import re
from decimal import Decimal
from pathlib import Path
from typing import List, Dict

import demjson
import requests
from bs4 import BeautifulSoup
from tqdm.auto import trange, tqdm


def download_speech_videos_and_transcripts(
        url: str,
        video_directory: Path,
        transcript_directory: Path
):
    speech_urls = get_speech_urls(url, n_pages=2)
    video_urls, video_transcripts = get_speech_video_urls_and_transcripts(speech_urls)

    video_directory.mkdir(parents=True, exist_ok=True)
    for i, video_url in enumerate(tqdm(video_urls, desc="Downloading speech videos", unit="video")):
        download_video(video_url, f"speech{i:02d}.mp4", video_directory)

    transcript_directory.mkdir(parents=True, exist_ok=True)
    for i, transcript in enumerate(tqdm(
            video_transcripts,
            desc="Downloading speech transcripts",
            unit="transcript"
    )):
        save_transcript(transcript, f"speech{i:02d}.json", transcript_directory)

    print(f"Saved speech videos to: {video_directory.resolve()}")
    print(f"Saved speech transcripts to: {transcript_directory.resolve()}")


def get_speech_urls(url: str, n_pages: int) -> List[str]:
    speech_urls = []
    for page_number in trange(n_pages, desc="Retrieving speech URLs", unit="page"):
        page_url = f"{url}/page/{page_number}"
        page = get_page_soup(page_url)
        grid = page.find("div", {"class": "fl-post-grid"})
        speech_urls += [a["href"] for a in grid.find_all("a")]
    return speech_urls


def get_speech_video_urls_and_transcripts(speech_urls: List[str]) -> (List[str], List[Dict]):
    video_urls = []
    video_transcripts = []
    for i, speech_url in enumerate(tqdm(
            speech_urls,
            desc="Retrieving speech metadata",
            unit="speech"
    )):
        video_json = get_speech_video_json(speech_url)
        video_url = video_json["model"]["mediaUrl"]
        video_urls.append(video_url)
        video_transcript = video_json["model"]["draft"]
        video_transcripts.append(video_transcript)
    return video_urls, video_transcripts


def download_video(url: str, filename: str, video_dir: Path):
    stream = requests.get(url, stream=True)
    total_size = int(stream.headers.get('content-length', 0))
    with tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            leave=False
    ) as progress_bar:
        filepath = video_dir / filename
        with filepath.open("wb") as file:
            for data in stream.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))


def save_transcript(transcript: Dict, filename: str, transcript_directory: Path):
    class DecimalEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Decimal):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    transcript_path = transcript_directory / filename
    with transcript_path.open("w") as transcript_file:
        json.dump(transcript, transcript_file, cls=DecimalEncoder)


def get_speech_video_json(speech_url: str) -> Dict:
    speech_page_soup = get_page_soup(speech_url)
    transcription = speech_page_soup.find("div", {"id": "transcription"})
    video_page_url = remove_url_params(transcription.find("a")["href"])
    video_page_content = get_page_content(video_page_url)
    match = re.search(r"var options = ((?:.|\n)+);", video_page_content)
    js_object = match.group(1)
    result = demjson.decode(js_object, return_errors=True)
    return result.object


def get_page_soup(url: str) -> BeautifulSoup:
    page_content = get_page_content(url)
    return BeautifulSoup(page_content, "html.parser")


def get_page_content(url: str) -> str:
    response = requests.get(url)
    assert response.status_code == 200
    return response.content.decode("utf-8")


def remove_url_params(url: str) -> str:
    return url.split('?')[0]
