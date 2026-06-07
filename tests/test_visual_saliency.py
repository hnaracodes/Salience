from __future__ import annotations

import numpy as np
from PIL import Image

from scout_core.visual_saliency import build_saliency_heatmap, score_snapshot_elements


def test_visual_saliency_scores_prominent_cta(tmp_path):
    frame = tmp_path / "frame.jpg"
    arr = np.full((100, 160, 3), 32, dtype=np.uint8)
    arr[20:50, 20:120] = [240, 180, 40]
    Image.fromarray(arr).save(frame)
    snapshot = {
        "scrollY": 0,
        "scrollX": 0,
        "elements": [
            {"dom_id": "#cta", "tag": "BUTTON", "text": "Get started", "bbox": [20, 20, 100, 30], "is_intersecting_viewport": True},
            {"dom_id": "#body", "tag": "P", "text": "Plain text", "bbox": [20, 70, 100, 18], "is_intersecting_viewport": True},
        ],
    }
    rows = score_snapshot_elements(frame, snapshot)
    assert rows[0]["dom_id"] == "#cta"
    heatmap, scored = build_saliency_heatmap(frame, snapshot, capture_h=100, capture_w=160)
    assert heatmap.shape == (100, 160)
    assert float(heatmap.max()) == 1.0
    assert scored[0]["dom_id"] == "#cta"
