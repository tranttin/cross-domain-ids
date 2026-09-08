from pathlib import Path
import pandas as pd

from ids.data.canonical_mapping import resolve_mapping, align_frames_with_mapping


def _cfg(tmp_path: Path):
    p = tmp_path / "map.yaml"
    p.write_text(
        '''version: test-v1
features:
  duration:
    mappings:
      A: {column: "Flow Duration", verified: true}
      B: {column: "Duration", verified: true}
  rate:
    mappings:
      A: {column: "Flow Packets/s", verified: true}
      B: {column: "Rate", verified: false}
''', encoding="utf-8")
    return p


def test_verified_mapping_excludes_candidate(tmp_path):
    p = _cfg(tmp_path)
    rm = resolve_mapping(["Flow Duration", "Flow Packets/s", "Label"], ["Duration", "Rate", "Label"], "A", "B", mapping_path=p, verified_only=True)
    assert [f.canonical for f in rm.features] == ["duration"]
    assert "rate" in rm.unverified


def test_align_frames_uses_canonical_order(tmp_path):
    p = _cfg(tmp_path)
    rm = resolve_mapping(["Flow Duration", "Flow Packets/s", "Label"], ["Duration", "Rate", "Label"], "A", "B", mapping_path=p, verified_only=False)
    a = pd.DataFrame({"Flow Duration": [1, 2], "Flow Packets/s": [3, 4], "Label": [0, 1]})
    b = pd.DataFrame({"Duration": [10, 20], "Rate": [30, 40], "Label": [1, 0]})
    aa, bb = align_frames_with_mapping(a, b, rm)
    assert aa.columns.tolist() == ["duration", "rate", "Label"]
    assert bb.columns.tolist() == ["duration", "rate", "Label"]
    assert bb["duration"].tolist() == [10, 20]
