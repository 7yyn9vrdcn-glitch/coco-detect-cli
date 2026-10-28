from __future__ import annotations

import argparse

import pytest

from coco_detect.regions import detect_regions, parse_region, region_fits

np = pytest.importorskip("numpy")


class _RecordingDetector:
    """Returns one box at crop (10,10)-(20,20); records each call."""

    def __init__(self):
        self.calls = []

    def detect_batch(self, images, imgsz=None):
        self.calls.append((imgsz, [img.shape[:2] for img in images]))
        return [[{"bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.5, "label": "sports ball"}]
                for _ in images]


def test_parse_rect_and_full():
    r = parse_region("0,600,1280,640@1280")
    assert (r.rect, r.imgsz, r.ymin) == ((0, 600, 1280, 640), 1280, None)
    f = parse_region("full@1280:ymin=1240")
    assert (f.rect, f.imgsz, f.ymin) == (None, 1280, 1240.0)


@pytest.mark.parametrize("raw", [
    "0,0,10,10", "0,0,10@640", "0,0,10,10@0", "-1,0,10,10@640",
    "0,0,10,10@640:ymin=5", "full@640:foo=1", "full@x",
])
def test_parse_rejects(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_region(raw)


def test_region_fits():
    assert region_fits(parse_region("2560,600,1280,640@1280"), 3840, 2160)
    assert not region_fits(parse_region("2600,600,1280,640@1280"), 3840, 2160)
    assert region_fits(parse_region("full@1280"), 3840, 2160)


def test_rect_boxes_are_mapped_back_to_source():
    image = np.zeros((2160, 3840, 3), dtype=np.uint8)
    det = _RecordingDetector()
    out = detect_regions(det, [image], [parse_region("1280,600,1280,640@1280")])
    assert det.calls == [(1280, [(640, 1280)])]
    assert out[0][0]["bbox"] == [1290.0, 610.0, 1300.0, 620.0]


def test_full_is_shrunk_scaled_back_and_filtered_by_ymin():
    image = np.zeros((2160, 3840, 3), dtype=np.uint8)
    det = _RecordingDetector()
    # crop box centre y=15 -> source y=15*3=45
    kept = detect_regions(det, [image], [parse_region("full@1280:ymin=40")])
    dropped = detect_regions(det, [image], [parse_region("full@1280:ymin=50")])
    assert det.calls[0] == (1280, [(720, 1280)])
    assert kept[0][0]["bbox"] == [30.0, 30.0, 60.0, 60.0]
    assert dropped == [[]]


def test_one_pass_per_imgsz_across_frames():
    images = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
    det = _RecordingDetector()
    out = detect_regions(det, images, [
        parse_region("0,0,640,360@640"), parse_region("640,0,640,360@640"),
        parse_region("full@960"),
    ])
    assert [c[0] for c in det.calls] == [640, 960]
    assert len(det.calls[0][1]) == 4  # 2 frames x 2 regions in one batch
    assert [len(d) for d in out] == [3, 3]
