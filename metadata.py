from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import cv2
from pprint import pprint
import numpy as np
import torch
import torchaudio


class Split(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class Sentiment(Enum):
    POSITIVE = 1
    NEUTRAL = 2
    NEGATIVE = 3


@dataclass
class Metadata:
    path: str
    video_id: str
    clip_id: int
    duration: float
    # VISUAL INFO
    width: int
    height: int
    # AUDIO INFO
    sample_rate: int
    channels: int
    # TEXT INFO
    text: str
    sentiment_score: float
    sentiment: Sentiment
    split: Split

    def get_cv2_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.path)

    def play_video(self):
        import IPython.display as ipd

        pprint(self)
        return ipd.Video(self.path)

    def get_video_frames(self, fps=10) -> np.ndarray:
        # note that due to the variable nature of FPS in encoded video,
        # the fps argument is only approximate for this function
        cap = self.get_cv2_capture()
        if not cap.isOpened():
            print("Error opening video file")

        frame_interval = 1.0 / fps  # interval in seconds

        frames = []
        last_capture_time = -frame_interval

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current time of the frame in the video
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # convert to seconds

            # Check if the current frame is due for capture based on desired FPS
            if current_time >= (last_capture_time + frame_interval):
                frames.append(frame)
                last_capture_time = current_time

        cap.release()
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        return np.array(frames)

    def get_audio(self) -> torch.Tensor:
        audio, sample_rate = torchaudio.load(self.path)
        return audio, sample_rate


class RawDataset:
    def __init__(
        self,
        data_dir: Path,
        dataset_name: str,
        max_workers: int = 8,
        debug: bool = False,
    ):
        self.dataset_dir = Path(data_dir) / dataset_name
        self.labels = pd.read_csv(self.dataset_dir / "label.csv")
        if debug:
            max_workers = 1
            self.labels = self.labels.sample(n=100)
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                splits = np.array_split(self.labels, max_workers)
                results = list(executor.map(self.build_video_sublist, splits))
            self.merge_lists(results)
        else:
            self.videos = self.build_video_sublist(self.labels)

    def merge_lists(self, results: list[dict, set]):
        # used for merging different results from different processors
        # if ProcessPoolExecutor is used
        self.videos = []
        for videos in results:
            self.videos += videos

    def read_video_metadata(self, video_path: Path, row) -> Metadata:
        # VIDEO METADATA
        vid = cv2.VideoCapture(filename=str(video_path))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # AUDIO METADATA
        audio, sample_rate = torchaudio.load(str(video_path), normalize=False)
        channels, audio_sample_count = audio.shape
        duration = audio_sample_count / sample_rate

        # LABELS
        if row.annotation == "Positive":
            sentiment = Sentiment.POSITIVE
        elif row.annotation == "Neutral":
            sentiment = Sentiment.NEUTRAL
        elif row.annotation == "Negative":
            sentiment = Sentiment.NEGATIVE
        else:
            raise ValueError(f"{row.annotation=} not recognized")

        # SPLIT
        if row.mode == "train":
            split = Split.TRAIN
        elif row.mode == "valid":
            split = Split.VALIDATION
        elif row.mode == "test":
            split = Split.TEST
        else:
            raise ValueError(f"{row.mode=} not recognized")

        return Metadata(
            path=str(video_path),
            duration=duration,
            width=width,
            height=height,
            sample_rate=sample_rate,
            channels=channels,
            video_id=row.video_id,
            clip_id=row.clip_id,
            text=row.text,
            sentiment_score=row.label,
            sentiment=sentiment,
            split=split,
        )

    def build_video_sublist(self, df: pd.DataFrame):
        # single processor code for reading metadata of videos from a Pandas DataFrame
        videos = []
        for row in df.itertuples():
            video_path = self.dataset_dir / "Raw" / row.video_id / f"{row.clip_id}.mp4"
            assert video_path.exists(), f"{video_path=} does not exist"
            videos.append(self.read_video_metadata(video_path=video_path, row=row))
        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, key):
        return self.videos[key]
