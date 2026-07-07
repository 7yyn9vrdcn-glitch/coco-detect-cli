from __future__ import annotations

from coco_detect.detector import YoloDetector


class _Scalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Vec:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return list(self._vals)


class _FakeBoxes:
    def __init__(self, rows):
        # rows: list of (cls_id, conf, xyxy)
        self.cls = [_Scalar(r[0]) for r in rows]
        self.conf = [_Scalar(r[1]) for r in rows]
        self.xyxy = [_Vec(r[2]) for r in rows]

    def __len__(self):
        return len(self.cls)


class _FakeResult:
    def __init__(self, rows, names):
        self.boxes = _FakeBoxes(rows)
        self.names = names


class _FakeModel:
    names = {0: "person", 32: "sports ball"}

    def __init__(self, per_call_results):
        self._per_call = list(per_call_results)
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append(images)
        return self._per_call.pop(0)


def _det(threshold=0.25, classes=None, results=None):
    model = _FakeModel(results or [])
    return YoloDetector("fake.pt", threshold, classes, model=model), model


def test_class_names_come_from_model_and_filter_applies():
    rows = [(0, 0.9, (1, 2, 3, 4)), (32, 0.5, (5, 6, 7, 8)), (16, 0.8, (0, 0, 1, 1))]
    det, _ = _det(
        classes={"person", "sports ball"},
        results=[[_FakeResult(rows, _FakeModel.names)]],
    )
    out = det.detect(object())
    assert [(d["label"], d["score"]) for d in out] == [("person", 0.9), ("sports ball", 0.5)]
    assert out[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_no_class_filter_keeps_everything():
    rows = [(16, 0.8, (0, 0, 1, 1))]
    det, _ = _det(classes=None, results=[[_FakeResult(rows, {16: "dog"})]])
    out = det.detect(object())
    assert out[0]["label"] == "dog"


def test_batch_preserves_order():
    r0 = _FakeResult([(0, 0.9, (0, 0, 1, 1))], _FakeModel.names)
    r1 = _FakeResult([(0, 0.8, (1, 1, 2, 2))], _FakeModel.names)
    det, _ = _det(results=[[r0, r1]])
    out = det.detect_batch([object(), object()])
    assert out[0][0]["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert out[1][0]["bbox"] == [1.0, 1.0, 2.0, 2.0]


def test_batch_count_mismatch_falls_back_to_per_frame():
    collapsed = [_FakeResult([(0, 0.9, (0, 0, 1, 1))], _FakeModel.names)]  # 1 result for 2 frames
    per0 = [_FakeResult([(0, 0.7, (2, 2, 3, 3))], _FakeModel.names)]
    per1 = [_FakeResult([], _FakeModel.names)]
    det, model = _det(results=[collapsed, per0, per1])
    out = det.detect_batch([object(), object()])
    assert len(out) == 2
    assert out[0][0]["score"] == 0.7
    assert out[1] == []
    assert len(model.calls) == 3  # batch + 2 per-frame fallbacks
