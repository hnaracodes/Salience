from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_validation_site_corpus.py"
SPEC = importlib.util.spec_from_file_location("run_validation_site_corpus", SCRIPT)
assert SPEC and SPEC.loader
corpus_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(corpus_runner)


def test_public_corpus_has_eight_distinct_sites():
    corpus = corpus_runner.load_corpus(ROOT / "configs" / "validation_site_corpus.yaml")
    sites = corpus["sites"]
    local_sites = corpus.get("local_sites") or []
    assert len(sites) == 8
    assert len(local_sites) >= 2
    assert len({site["id"] for site in sites}) == 8
    assert all(str(site["url"]).startswith("https://") for site in sites)
    assert all(str(site["url"]).startswith("http://127.0.0.1") for site in local_sites)


def test_full_command_forwards_target_url_and_norm():
    corpus = corpus_runner.load_corpus(ROOT / "configs" / "validation_site_corpus.yaml")
    site = corpus["sites"][0]
    cmd = corpus_runner.build_command(
        site=site,
        corpus=corpus,
        session_id="validation_test",
        stage="all",
    )
    assert cmd[cmd.index("--url") + 1] == site["url"]
    assert cmd[cmd.index("--session-id") + 1] == "validation_test"
    assert cmd[cmd.index("--norm-id") + 1] == "synthetic_bootstrap_v1"
    assert "--skip-narrative" in cmd


def test_local_command_uses_script_without_url_injection():
    corpus = corpus_runner.load_corpus(ROOT / "configs" / "validation_site_corpus.yaml")
    site = corpus["local_sites"][0]
    cmd = corpus_runner.build_command(
        site=site,
        corpus=corpus,
        session_id="validation_local",
        stage="capture",
    )
    assert "--url" not in cmd
    assert site["script"].replace("\\", "/") in "/".join(cmd).replace("\\", "/")


def test_capture_command_does_not_require_norm():
    corpus = corpus_runner.load_corpus(ROOT / "configs" / "validation_site_corpus.yaml")
    cmd = corpus_runner.build_command(
        site=corpus["sites"][0],
        corpus=corpus,
        session_id="validation_test",
        stage="capture",
    )
    assert "--norm-id" not in cmd
