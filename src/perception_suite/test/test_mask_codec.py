"""The mask codec round trip.

A mask carries one boolean per pixel plus a single frame-wide confidence, so it
is packed into mono8 with the confidence as the pixel level. The edge cases are
what make that safe: 0 must mean "not road", and a real but tiny confidence must
not round to 0 and take the whole mask with it.
"""

import numpy as np

from perception_suite.common import decode_mask, encode_mask


def _mask():
    m = np.zeros((8, 8), dtype=bool)
    m[2:6, 3:7] = True
    return m


def test_round_trip_preserves_the_mask():
    m = _mask()
    out, _ = decode_mask(encode_mask(m, 0.75))
    assert (out == m).all()


def test_round_trip_preserves_confidence_within_quantisation():
    for conf in (0.05, 0.25, 0.5, 0.75, 1.0):
        _, out = decode_mask(encode_mask(_mask(), conf))
        assert abs(out - conf) <= 1.0 / 255.0


def test_tiny_confidence_does_not_blank_the_mask():
    # 0.001 * 255 rounds to 0. Without the clamp the mask would decode empty and
    # the planner would see no road at all rather than low-confidence road.
    m = _mask()
    out, conf = decode_mask(encode_mask(m, 0.001))
    assert (out == m).all()
    assert conf > 0.0


def test_zero_confidence_is_empty():
    out, conf = decode_mask(encode_mask(_mask(), 0.0))
    assert not out.any()
    assert conf == 0.0


def test_empty_mask_decodes_empty():
    out, conf = decode_mask(encode_mask(np.zeros((8, 8), dtype=bool), 0.9))
    assert not out.any()
    assert conf == 0.0


def test_encoded_image_is_mono8_shaped():
    enc = encode_mask(_mask(), 0.5)
    assert enc.dtype == np.uint8
    assert enc.shape == (8, 8)
