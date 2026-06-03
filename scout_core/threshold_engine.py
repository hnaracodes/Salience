from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from scout_core.constants import NETWORK_NAME_TO_ID
from scout_core.schemas import ThresholdContext, ThresholdHit


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _network_column_index(ctx: ThresholdContext, network_name: str) -> int:
    target = network_name.strip()
    if target in ctx.network_names:
        return ctx.network_names.index(target)
    # Prefix match for coarse keys against subnetwork-style names (legacy bundles).
    for idx, name in enumerate(ctx.network_names):
        if name == target or name.startswith(f"{target}_") or target.startswith(f"{name}_"):
            return idx
    nid = NETWORK_NAME_TO_ID.get(target)
    if nid is not None:
        from scout_core.constants import network_names_for_ids

        alt = network_names_for_ids([nid])[0]
        if alt in ctx.network_names:
            return ctx.network_names.index(alt)
    raise KeyError(f"Unknown network {network_name!r}; known={ctx.network_names}")


def _runs_from_mask(mask: list[bool]) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def evaluate_rules(
    ctx: ThresholdContext,
    rules_path: Path,
    insight_catalog_path: Path,
) -> list[ThresholdHit]:
    rules_doc = _load_yaml(rules_path)
    catalog = _load_yaml(insight_catalog_path)
    insights = catalog.get("insights", {})

    z = ctx.z_network
    T = len(z)
    fps = ctx.fps or 1.0
    hits: list[ThresholdHit] = []

    for rule in rules_doc.get("rules", []):
        rid = rule["id"]
        when = rule["when"]
        net_name = when["network"]
        z_above = float(when.get("z_above", 0.0))
        min_dur_s = float(when.get("min_duration_s", rules_doc.get("windows", {}).get("spike_min_seconds", 0.5)))
        min_frames = max(1, int(math.ceil(min_dur_s * fps)))

        col = _network_column_index(ctx, net_name)
        mask = [bool(z[t][col] > z_above) for t in range(T)]
        for t0, t1 in _runs_from_mask(mask):
            dur = t1 - t0 + 1
            if dur < min_frames:
                continue
            window_z = [z[t][col] for t in range(t0, t1 + 1)]
            excess = sum(max(0.0, float(zv) - z_above) for zv in window_z)
            confidence = float(excess * (dur / max(fps, 1e-6)))
            ikey = rule.get("insight_key", rid)
            itext = insights.get(ikey, {}).get("text", f"Rule {rid} matched.")

            evidence = {
                "network": net_name,
                "z_above": z_above,
                "mean_z_window": sum(window_z) / len(window_z),
                "min_duration_frames": min_frames,
                "fps": fps,
            }

            hits.append(
                ThresholdHit(
                    session_id=ctx.session_id,
                    rule_id=rid,
                    t_start=t0,
                    t_end=t1,
                    networks_json=json.dumps([net_name]),
                    evidence_json=json.dumps(evidence),
                    confidence=confidence,
                    insight_key=ikey,
                    insight_text=str(itext),
                )
            )

    return hits
