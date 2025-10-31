from __future__ import annotations

import io
from typing import Tuple

import librosa
import numpy as np


TARGET_SR = 16000


def load_audio_bytes(file_bytes: bytes, sr: int = TARGET_SR, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio from raw bytes into a numpy waveform at target sampling rate.
    """
    with io.BytesIO(file_bytes) as bio:
        y, s = librosa.load(bio, sr=sr, mono=mono)
    return y, s


def compute_mfcc(y: np.ndarray, sr: int = TARGET_SR, n_mfcc: int = 40) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def compute_melspectrogram(y: np.ndarray, sr: int = TARGET_SR, n_mels: int = 64) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db
