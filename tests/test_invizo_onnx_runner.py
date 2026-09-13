import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))
from invizo_onnx_run import CheckedSession, checked_predict


def test_upstream_swallowed_error_cannot_be_written_as_prediction():
    def fail(*args):
        raise ValueError("ONNX shape mismatch")
    session = CheckedSession(SimpleNamespace(run=fail))
    def upstream_predict(image):
        try:
            session.run(None, {})
        except Exception:
            pass
        return "partial"
    with pytest.raises(RuntimeError, match="partial prediction was discarded"):
        checked_predict(SimpleNamespace(predict=upstream_predict), session, None)


def test_success_preserves_upstream_text():
    session = CheckedSession(None)
    assert checked_predict(SimpleNamespace(predict=lambda image: "كتب"), session, None) == "كتب"
