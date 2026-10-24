
import hashlib
import json
import numpy as np

MIN_RECORDS = 64
BINS = 64

def image_histogram(raw_images):
    pixels = np.asarray(raw_images)
    if pixels.ndim != 4 or pixels.dtype != np.uint8:
        raise ValueError('Expected raw uint8 records in NCHW order')
    channels = pixels.shape[1]
    if channels not in (1, 3):
        raise ValueError('Expected one or three image channels')
    return np.stack([np.bincount((pixels[:, c].ravel() // 4), minlength=BINS)
                     for c in range(channels)]).ravel().astype(np.int64)

def perturb(hist, rng, nominal_scale=1.0, max_abs=8):

    noise = np.rint(rng.laplace(0, nominal_scale, len(hist))).astype(np.int64)
    return hist + np.clip(noise, -max_abs, max_abs), np.clip(noise, -max_abs, max_abs)

def anchor_commitment(public_key, anchor):
    values = np.asarray(anchor)
    coords = [float(v) for v in values] if np.issubdtype(values.dtype, np.floating) else [int(v) for v in values]
    fields = {'key': str(public_key), 'anchor': coords}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def verify_image_relation(raw_images, anchor, noise, public_key, expected_commitment,
                          min_records=MIN_RECORDS, max_abs=8):
    pixels = np.asarray(raw_images)
    if pixels.ndim != 4 or pixels.shape[0] < min_records:
        return False
    try:
        hist = image_histogram(pixels)
    except ValueError:
        return False
    anchor, noise = np.asarray(anchor), np.asarray(noise)
    return bool(anchor.shape == hist.shape and noise.shape == hist.shape and
                np.issubdtype(anchor.dtype, np.integer) and
                np.issubdtype(noise.dtype, np.integer) and
                np.all(np.abs(noise) <= max_abs) and np.array_equal(anchor, hist + noise) and
                expected_commitment == anchor_commitment(public_key, anchor))
