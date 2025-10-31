import numpy as np
from app.services.audio_processing import compute_mfcc


def test_compute_mfcc_shape():
    sr = 16000
    # 1 second of silence
    y = np.zeros(sr, dtype=np.float32)
    mfcc = compute_mfcc(y, sr=sr, n_mfcc=20)
    # shape: (n_mfcc, frames)
    assert mfcc.shape[0] == 20
    assert mfcc.shape[1] > 0
