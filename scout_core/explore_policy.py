"""Bounded heuristic policy for autonomous website exploration (scroll_mode: explore)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import yaml

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPLORE_CONFIG = PROJECT_ROOT / "configs" / "explore_defaults.yaml"

CTA_TERMS = (
    "get started",
    "start",
    "try",
    "demo",
    "sign up",
    "subscribe",
    "pricing",
    "buy",
    "contact",
    "learn more",
    "free trial",
)

DENY_HREF_SUBSTR = ("logout", "delete", "checkout", "payment", "signout", "sign-out")


@dataclass
class ExploreBudget:
    max_tr: int = 24
    max_clicks: int = 12
    max_pages: int = 6
    t_idx: int = 0
    clicks: int = 0
    pages_visited: list[str] = field(default_factory=list)
    visited_actions: set[str] = field(default_factory=set)

    def record_page(self, url: str) -> None:
        path = urlparse(url).path or "/"
        if path not in self.pages_visited:
            self.pages_visited.append(path)

    def record_click(self, action_key: str) -> None:
        self.clicks += 1
        self.visited_actions.add(action_key)

    def at_limit(self) -> bool:
        return (
            self.t_idx >= self.max_tr
            or self.clicks >= self.max_clicks
            or len(self.pages_visited) >= self.max_pages
        )


@dataclass
class ExploreAction:
    kind: str  # scroll_down | click | hover | wait | stop
    selector: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_explore_config(script: dict[str, Any], defaults_path: Path | None = None) -> dict[str, Any]:
    """Merge script explore: block with explore_defaults.yaml."""
    cfg_path = defaults_path or DEFAULT_EXPLORE_CONFIG
    base: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        base = dict(raw.get("explore") or {})
    override = script.get("explore") or {}
    base.update(override)
    return base


def is_same_origin(href: str, initial_url: str, cfg: dict[str, Any]) -> bool:
    if not href or href.startswith("#"):
        return False
    if cfg.get("same_origin_only", True) is False:
        return True
    try:
        base = urlparse(initial_url)
        target = urlparse(urljoin(initial_url, href))
    except Exception:
        return False
    if target.scheme not in ("http", "https", ""):
        return False
    if cfg.get("allow_subdomains"):
        return (
            target.netloc == base.netloc
            or target.netloc.endswith("." + base.netloc)
        )
    return target.netloc == base.netloc


def _href_denied(href: str, cfg: dict[str, Any]) -> bool:
    href_l = (href or "").lower()
    for sub in DENY_HREF_SUBSTR:
        if sub in href_l:
            return True
    for pat in cfg.get("deny_href_patterns") or []:
        if re.search(pat, href_l, re.I):
            return True
    return False


def _selector_denied(selector: str, cfg: dict[str, Any]) -> bool:
    sel_l = (selector or "").lower()
    for pat in cfg.get("deny_selectors") or []:
        if pat.lower() in sel_l:
            return True
    return False


def _position_score(bbox: list[int], viewport_h: int) -> float:
    if len(bbox) < 4 or viewport_h <= 0:
        return 0.5
    cy = bbox[1] + bbox[3] / 2.0
    return float(max(0.0, 1.0 - cy / viewport_h))


def _href_path(href: str, initial_url: str) -> str:
    if not href or href.startswith("#"):
        return ""
    try:
        target = urlparse(urljoin(initial_url, href))
    except Exception:
        return ""
    return target.path or "/"


def element_selector(el: dict[str, Any], *, initial_url: str = "") -> str:
    """Map a DOM snapshot row to a Playwright-safe CSS selector."""
    dom_id = str(el.get("dom_id") or "").strip()
    href = str(el.get("href") or "").strip()
    tag = (el.get("tag") or "").upper()

    if dom_id.startswith(("#", ".", "[")):
        return dom_id

    # Bare tag names (e.g. "a") collide; disambiguate with href when possible.
    if dom_id in ("a", "button") or (tag == "A" and dom_id == "a"):
        if href:
            esc = href.replace("\\", "\\\\").replace('"', '\\"')
            return f'a[href="{esc}"]'
        return "a"

    if dom_id:
        # tag.class forms (e.g. a.nav-cta) are valid selectors as-is.
        if dom_id[0].isalpha() and ("." in dom_id or dom_id.isalpha()):
            return dom_id
        return f"#{dom_id.lstrip('#')}"
    return ""


def _scroll_fraction(scroll_y: int, scroll_height: int | None, client_height: int | None) -> float:
    if scroll_height is None or client_height is None:
        return 1.0
    max_scroll = max(0, int(scroll_height) - int(client_height))
    if max_scroll <= 0:
        return 1.0
    return float(scroll_y) / float(max_scroll)


def _clicks_allowed(
    *,
    scroll_y: int,
    scroll_height: int | None,
    client_height: int | None,
    cfg: dict[str, Any],
) -> bool:
    if not cfg.get("scroll_to_bottom_before_clicks", True):
        return True
    frac = _scroll_fraction(scroll_y, scroll_height, client_height)
    threshold = float(cfg.get("min_scroll_fraction_before_clicks", 0.92))
    max_scroll = 0
    if scroll_height is not None and client_height is not None:
        max_scroll = max(0, int(scroll_height) - int(client_height))
    at_bottom = scroll_y >= max_scroll if max_scroll > 0 else True
    return at_bottom or frac >= threshold


def score_actionable_elements(
    elements: list[dict[str, Any]],
    *,
    page_url: str,
    initial_url: str,
    budget: ExploreBudget,
    cfg: dict[str, Any],
    viewport_h: int = 720,
) -> list[tuple[float, dict[str, Any], str]]:
    """Return ranked (score, element, action_kind) candidates."""
    ranked: list[tuple[float, dict[str, Any], str]] = []
    for el in elements or []:
        if not el.get("is_intersecting_viewport", True):
            continue
        dom_id = str(el.get("dom_id") or "")
        tag = (el.get("tag") or "").upper()
        role = (el.get("role") or "").lower()
        text = (el.get("text") or "").lower()
        href = str(el.get("href") or "")
        selector = element_selector(el, initial_url=initial_url)
        if not selector or _selector_denied(selector, cfg):
            continue
        action_key = f"{page_url}|{selector}"
        if action_key in budget.visited_actions:
            continue

        score = 0.0
        action_kind = "click"

        if tag == "A" and href:
            if not is_same_origin(href, initial_url, cfg) or _href_denied(href, cfg):
                continue
            score += 0.55
            if "nav" in dom_id.lower() or role == "navigation":
                score += 0.15
            if cfg.get("prioritize_unvisited_nav", True):
                path = _href_path(href, initial_url)
                if path and path not in budget.pages_visited:
                    score += float(cfg.get("unvisited_page_bonus", 0.4))
            action_kind = "click"
        elif tag == "BUTTON" or role == "button" or "cta" in dom_id.lower() or el.get("data-cta"):
            score += 0.7
            action_kind = "click"
        else:
            continue

        for term in CTA_TERMS:
            if term in text or term in dom_id.lower():
                score += 0.25
                break

        score += 0.2 * _position_score(el.get("bbox") or [0, 0, 0, 0], viewport_h)
        vis = float(el.get("visibility_ratio") or 1.0)
        score *= max(0.2, vis)
        ranked.append((score, el, action_kind))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def next_action(
    *,
    snapshot: dict[str, Any],
    page_url: str,
    initial_url: str,
    budget: ExploreBudget,
    cfg: dict[str, Any],
    viewport_h: int = 720,
    scroll_height: int | None = None,
    client_height: int | None = None,
) -> ExploreAction:
    """Decide the next exploration action at the current TR."""
    if budget.at_limit():
        return ExploreAction(kind="stop", reason="budget_exhausted")

    elements = snapshot.get("elements") or []
    scroll_y = int(snapshot.get("scrollY", 0))
    ranked = score_actionable_elements(
        elements,
        page_url=page_url,
        initial_url=initial_url,
        budget=budget,
        cfg=cfg,
        viewport_h=viewport_h,
    )
    can_click = _clicks_allowed(
        scroll_y=scroll_y,
        scroll_height=scroll_height,
        client_height=client_height,
        cfg=cfg,
    )
    click_threshold = float(cfg.get("click_score_threshold", 0.5))
    unvisited_threshold = float(cfg.get("unvisited_click_threshold", 0.35))
    current_path = urlparse(page_url).path or "/"

    if ranked and budget.clicks < budget.max_clicks:
        score, el, kind = ranked[0]
        dom_id = str(el.get("dom_id") or "")
        selector = element_selector(el, initial_url=initial_url)
        href = str(el.get("href") or "")
        path = _href_path(href, initial_url) if href else ""
        is_unvisited_nav = bool(
            path and path not in budget.pages_visited and path != current_path
        )
        allow_now = can_click or (
            is_unvisited_nav and cfg.get("allow_nav_clicks_during_scroll", True)
        )
        threshold = unvisited_threshold if is_unvisited_nav else click_threshold
        if allow_now and score >= threshold and selector:
            return ExploreAction(
                kind=kind,
                selector=selector,
                reason=f"element_score={score:.2f}",
                metadata={"dom_id": dom_id, "score": round(score, 4), "href": href},
            )

    scroll_px = int(cfg.get("scroll_px_per_tr", 540))
    max_scroll = 0
    if scroll_height is not None and client_height is not None:
        max_scroll = max(0, int(scroll_height) - int(client_height))
    if scroll_y < max_scroll:
        return ExploreAction(
            kind="scroll_down",
            reason="below_fold_content",
            metadata={"scroll_px": scroll_px, "scroll_y": scroll_y},
        )

    if ranked and budget.clicks < budget.max_clicks:
        score, el, kind = ranked[0]
        dom_id = str(el.get("dom_id") or "")
        selector = element_selector(el, initial_url=initial_url)
        href = str(el.get("href") or "")
        path = _href_path(href, initial_url) if href else ""
        if selector and (path not in budget.pages_visited or score >= unvisited_threshold):
            return ExploreAction(
                kind=kind,
                selector=selector,
                reason=f"fallback_click score={score:.2f}",
                metadata={"dom_id": dom_id, "href": href},
            )

    return ExploreAction(kind="stop", reason="page_end_no_actions")
