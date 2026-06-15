"""Unit tests for explore_policy bounded autonomous capture."""

from __future__ import annotations

from scout_core.explore_policy import (
    ExploreBudget,
    element_selector,
    is_same_origin,
    load_explore_config,
    next_action,
    safe_locator_strategy,
    score_actionable_elements,
)


def test_cta_outranks_body_link():
    elements = [
        {
            "dom_id": "#copy",
            "tag": "P",
            "text": "Some body copy",
            "bbox": [20, 400, 400, 40],
            "is_intersecting_viewport": True,
            "visibility_ratio": 1.0,
        },
        {
            "dom_id": "#cta-hero",
            "tag": "BUTTON",
            "text": "Get started free",
            "bbox": [20, 100, 160, 48],
            "is_intersecting_viewport": True,
            "visibility_ratio": 1.0,
        },
    ]
    budget = ExploreBudget()
    cfg = {"same_origin_only": True}
    ranked = score_actionable_elements(
        elements,
        page_url="http://127.0.0.1:8780/",
        initial_url="http://127.0.0.1:8780/index.html",
        budget=budget,
        cfg=cfg,
    )
    assert ranked
    assert ranked[0][1]["dom_id"] == "#cta-hero"


def test_external_href_blocked():
    elements = [
        {
            "dom_id": "#ext",
            "tag": "A",
            "text": "External",
            "href": "https://evil.example.com/page",
            "bbox": [10, 10, 80, 20],
            "is_intersecting_viewport": True,
        },
    ]
    budget = ExploreBudget()
    ranked = score_actionable_elements(
        elements,
        page_url="http://127.0.0.1:8780/",
        initial_url="http://127.0.0.1:8780/index.html",
        budget=budget,
        cfg={"same_origin_only": True},
    )
    assert ranked == []


def test_budget_stop():
    budget = ExploreBudget(max_tr=2, max_clicks=1, max_pages=2)
    budget.t_idx = 2
    action = next_action(
        snapshot={"elements": [], "scrollY": 0},
        page_url="http://127.0.0.1:8780/",
        initial_url="http://127.0.0.1:8780/",
        budget=budget,
        cfg={},
    )
    assert action.kind == "stop"


def test_is_same_origin_relative_href():
    cfg = {"same_origin_only": True}
    assert is_same_origin("/pricing.html", "http://127.0.0.1:8780/index.html", cfg)
    assert is_same_origin("features.html", "http://127.0.0.1:8780/index.html", cfg)
    assert not is_same_origin("https://other.com/x", "http://127.0.0.1:8780/index.html", cfg)


def test_load_explore_config_merges_script():
    script = {"explore": {"max_tr": 10}}
    cfg = load_explore_config(script)
    assert cfg["max_tr"] == 10
    assert "max_clicks" in cfg


def test_element_selector_uses_href_for_bare_anchor():
    el = {"dom_id": "a", "tag": "A", "href": "features.html"}
    assert element_selector(el) == 'a[href="features.html"]'


def test_unvisited_nav_outranks_footer_cta_at_page_end():
    elements = [
        {
            "dom_id": "#cta-primary",
            "tag": "A",
            "text": "Start free trial",
            "href": "pricing.html",
            "bbox": [20, 2000, 160, 48],
            "is_intersecting_viewport": True,
            "visibility_ratio": 1.0,
        },
        {
            "dom_id": "a",
            "tag": "A",
            "text": "Features",
            "href": "features.html",
            "bbox": [120, 20, 80, 20],
            "is_intersecting_viewport": True,
            "visibility_ratio": 1.0,
        },
    ]
    budget = ExploreBudget()
    budget.record_page("/index.html")
    budget.pages_visited.append("/pricing.html")
    cfg = {
        "same_origin_only": True,
        "prioritize_unvisited_nav": True,
        "unvisited_page_bonus": 0.4,
    }
    ranked = score_actionable_elements(
        elements,
        page_url="http://127.0.0.1:8780/index.html",
        initial_url="http://127.0.0.1:8780/index.html",
        budget=budget,
        cfg=cfg,
    )
    assert ranked
    assert ranked[0][1]["href"] == "features.html"


def test_scroll_before_click_blocks_early_cta():
    elements = [
        {
            "dom_id": "#hero-cta",
            "tag": "BUTTON",
            "text": "Start free trial",
            "href": "",
            "bbox": [20, 100, 160, 48],
            "is_intersecting_viewport": True,
            "visibility_ratio": 1.0,
        },
    ]
    budget = ExploreBudget()
    action = next_action(
        snapshot={"elements": elements, "scrollY": 200},
        page_url="http://127.0.0.1:8780/index.html",
        initial_url="http://127.0.0.1:8780/index.html",
        budget=budget,
        cfg={
            "scroll_to_bottom_before_clicks": True,
            "min_scroll_fraction_before_clicks": 0.92,
            "allow_nav_clicks_during_scroll": True,
        },
        scroll_height=4000,
        client_height=720,
    )
    assert action.kind == "scroll_down"


def test_safe_locator_strategy_link_returns_role_and_name():
    el = {
        "tag": "A",
        "text": "Get started free",
        "href": "/pricing",
    }
    result = safe_locator_strategy(el, allow_role_locators=True)
    assert result["locator_role"] == "link"
    assert result["locator_name"] == "Get started free"
    assert result["locator_href"] == "/pricing"
    assert "selector" not in result


def test_safe_locator_strategy_blocks_css_in_production_mode():
    """Production mode must not emit a raw CSS selector."""
    el = {
        "tag": "BUTTON",
        "dom_id": "#hero-cta.btn.btn-primary.cta-dangerous[onclick='eval(x)']",
        "text": "Click me",
        "href": "",
    }
    result = safe_locator_strategy(el, allow_role_locators=True)
    # Role-based result should contain no CSS/XPath
    assert "selector" not in result
    assert result["locator_role"] == "button"
    assert result["locator_name"] == "Click me"


def test_safe_locator_strategy_legacy_mode_returns_selector():
    el = {
        "tag": "A",
        "dom_id": "#nav-link",
        "text": "Features",
        "href": "/features",
    }
    result = safe_locator_strategy(el, allow_role_locators=False)
    assert "selector" in result
    assert "locator_role" not in result
