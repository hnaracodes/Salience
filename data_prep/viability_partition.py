from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from data_prep.schemas import M0NetworkResult, M0Report
from scout_core.constants import NETWORK_NAME_TO_ID, YEO7_NAMES

# Networks of interest for go criterion (Vis, Default, SalVentAttn)
PRIORITY_NETWORKS = ("Vis", "Default", "SalVentAttn")

NET_COLUMN_PREFIX = "net_"


def age_to_band(age: float) -> str:
    if age < 18:
        raise ValueError(f"age must be >= 18, got {age}")
    if age <= 35:
        return "young_adult"
    if age <= 55:
        return "middle_adult"
    return "older_adult"


def _one_hot(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.ones((len(df), 1))
    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(df[columns].astype(str))
    # Single category (e.g. one site in HCP-only pilot) → intercept-only design matrix.
    if X.shape[1] == 0:
        return np.ones((len(df), 1))
    return X


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _fit_r2(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[0] < 2 or X.shape[1] < 1:
        return 0.0
    reg = LinearRegression()
    reg.fit(X, y)
    return _r2(y, reg.predict(X))


def _demographic_r2_given_site(
    y: np.ndarray,
    age_band: pd.Series,
    sex: pd.Series,
    site_id: pd.Series,
) -> tuple[float, float, float]:
    """Return (r2_demo_given_site, r2_site_only, r2_demo_only)."""
    meta = pd.DataFrame({"age_band": age_band, "sex": sex, "site_id": site_id})
    X_site = _one_hot(meta, ["site_id"])
    X_demo = _one_hot(meta, ["age_band", "sex"])
    X_full = np.hstack([X_site, X_demo])

    r2_site = _fit_r2(X_site, y)
    r2_demo = _fit_r2(X_demo, y)
    r2_full = _fit_r2(X_full, y)
    r2_demo_given_site = max(0.0, r2_full - r2_site)
    return r2_demo_given_site, r2_site, r2_demo


def _permute_demographics_within_site(
    age_band: pd.Series,
    sex: pd.Series,
    site_id: pd.Series,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    age_out = age_band.copy()
    sex_out = sex.copy()
    for site in site_id.unique():
        mask = site_id == site
        idx = np.where(mask)[0]
        if idx.size < 2:
            continue
        perm = rng.permutation(idx.size)
        age_out.iloc[idx] = age_band.iloc[idx].values[perm]
        sex_out.iloc[idx] = sex.iloc[idx].values[perm]
    return age_out, sex_out


def network_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(NET_COLUMN_PREFIX)]
    if cols:
        return sorted(cols)
    # Allow net_1 .. net_7
    numeric = [f"net_{i}" for i in range(1, 8) if f"net_{i}" in df.columns]
    return numeric


def network_name_from_column(col: str) -> str:
    suffix = col.removeprefix(NET_COLUMN_PREFIX)
    if suffix in NETWORK_NAME_TO_ID:
        return suffix
    try:
        idx = int(suffix) - 1
        if 0 <= idx < len(YEO7_NAMES):
            return YEO7_NAMES[idx]
    except ValueError:
        pass
    return suffix


def analyze_network(
    df: pd.DataFrame,
    net_col: str,
    *,
    n_permutations: int,
    random_state: int,
) -> M0NetworkResult:
    required = {"subject_id", "site_id", "age_band", "sex", net_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for network analysis: {sorted(missing)}")

    y = df[net_col].astype(np.float64).values
    age_band = df["age_band"]
    sex = df["sex"]
    site_id = df["site_id"]

    r2_demo_site, r2_site, r2_demo = _demographic_r2_given_site(y, age_band, sex, site_id)

    rng = np.random.default_rng(random_state)
    null_stats: list[float] = []
    for _ in range(n_permutations):
        p_age, p_sex = _permute_demographics_within_site(age_band, sex, site_id, rng)
        r2_perm, _, _ = _demographic_r2_given_site(y, p_age, p_sex, site_id)
        null_stats.append(r2_perm)

    null_arr = np.asarray(null_stats, dtype=np.float64)
    p_val = float((null_arr >= r2_demo_site).sum() + 1) / (n_permutations + 1)
    null_95 = float(np.percentile(null_arr, 95)) if null_arr.size else 0.0

    net_name = network_name_from_column(net_col)
    net_id = NETWORK_NAME_TO_ID.get(net_name, 0)

    return M0NetworkResult(
        network_name=net_name,
        network_id=net_id,
        r2_demographic_given_site=r2_demo_site,
        r2_site_only=r2_site,
        r2_demographic_only=r2_demo,
        permutation_p=p_val,
        permutation_null_95=null_95,
        n_subjects=len(df),
    )


def evaluate_go_criteria(results: list[M0NetworkResult]) -> dict[str, bool]:
    by_name = {r.network_name: r for r in results}

    priority_hits = [
        r
        for name in PRIORITY_NETWORKS
        if (r := by_name.get(name)) is not None and r.r2_demographic_given_site >= 0.01
    ]
    criterion_variance = len(priority_hits) >= 2

    criterion_permutation = sum(
        1 for name in PRIORITY_NETWORKS if (r := by_name.get(name)) and r.permutation_p < 0.05
    ) >= 2

    criterion_beats_site = any(
        (r := by_name.get(name))
        and r.r2_demographic_given_site > 0.0
        and r.r2_demographic_only > r.r2_site_only
        for name in PRIORITY_NETWORKS
    )

    return {
        "variance_in_two_priority_networks": criterion_variance,
        "permutation_significant_in_two_networks": criterion_permutation,
        "demographic_beats_site_in_one_network": criterion_beats_site,
    }


def _aggregate_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject_id; mean network features across duplicate rows."""
    if df["subject_id"].duplicated().any():
        net_cols = network_columns(df)
        meta_cols = [c for c in df.columns if c not in net_cols]
        agg: dict[str, str | type] = {c: "first" for c in meta_cols if c != "subject_id"}
        for c in net_cols:
            agg[c] = "mean"
        df = df.groupby("subject_id", as_index=False).agg(agg)
    return df


def run_viability_analysis(
    df: pd.DataFrame,
    *,
    data_source: str,
    is_synthetic: bool = False,
    multi_site_validated: bool | None = None,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> M0Report:
    df = _aggregate_subjects(df.copy())
    net_cols = network_columns(df)
    if not net_cols:
        raise ValueError(
            f"No network columns found (expected {NET_COLUMN_PREFIX}<name> or net_1..net_7)"
        )

    results = [
        analyze_network(
            df,
            col,
            n_permutations=n_permutations,
            random_state=random_state + i,
        )
        for i, col in enumerate(net_cols)
    ]

    go = evaluate_go_criteria(results)
    passed = all(go.values())
    n_sites = int(df["site_id"].astype(str).nunique()) if "site_id" in df.columns else 1
    if multi_site_validated is None:
        multi_site_validated = n_sites >= 2
    notes: list[str] = [
        "OLS incremental R² on subject-level network aggregates (not mixed model)."
    ]
    if is_synthetic:
        notes.append(
            "Synthetic data — passed flag validates pipeline logic only, not scientific viability."
        )
        passed = False
        multi_site_validated = False
    if n_sites < 2:
        notes.append("single_site — not a production M0 gate; add Cam-CAN or second dataset.")
        multi_site_validated = False

    return M0Report(
        passed=passed,
        is_synthetic=is_synthetic,
        multi_site_validated=multi_site_validated,
        n_sites=n_sites,
        data_source=data_source,
        n_subjects=len(df),
        n_permutations=n_permutations,
        networks=results,
        go_criteria=go,
        notes=notes,
    )


def generate_synthetic_viability_df(
    n_subjects: int = 120,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Synthetic subjects with detectable demographic signal on Vis and Default."""
    rng = np.random.default_rng(random_state)
    sites = ["site_a", "site_b"]
    ages = rng.integers(22, 65, size=n_subjects)
    sexes = rng.choice(["female", "male"], size=n_subjects)
    datasets = rng.choice(["hcp_7t", "camcan"], size=n_subjects)
    rows: list[dict[str, Any]] = []

    for i in range(n_subjects):
        age_band = age_to_band(float(ages[i]))
        sex = str(sexes[i])
        site = sites[i % len(sites)]
        demo_effect_vis = 0.15 if sex == "female" else -0.05
        demo_effect_default = 0.1 if age_band == "older_adult" else 0.0
        site_effect = 0.3 if site == "site_b" else 0.0

        row: dict[str, Any] = {
            "subject_id": f"sub-{i:04d}",
            "site_id": site,
            "dataset": str(datasets[i]),
            "age": float(ages[i]),
            "age_band": age_band,
            "sex": sex,
        }
        for j, name in enumerate(YEO7_NAMES):
            base = rng.normal(0.0, 0.5)
            if name == "Vis":
                base += demo_effect_vis + site_effect * 0.5
            elif name == "Default":
                base += demo_effect_default + site_effect * 0.3
            elif name == "SalVentAttn":
                base += (0.08 if sex == "male" else 0.0) + site_effect * 0.2
            else:
                base += site_effect * 0.1
            row[f"{NET_COLUMN_PREFIX}{name}"] = base + rng.normal(0, 0.1)
        rows.append(row)

    return pd.DataFrame(rows)


def load_subjects_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject_id", "site_id", "age_band", "sex"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("CSV is empty")
    allowed_bands = {"young_adult", "middle_adult", "older_adult"}
    bad_bands = set(df["age_band"].astype(str).unique()) - allowed_bands
    if bad_bands:
        raise ValueError(f"Invalid age_band values: {sorted(bad_bands)}")
    allowed_sex = {"female", "male", "unknown"}
    bad_sex = set(df["sex"].astype(str).unique()) - allowed_sex
    if bad_sex:
        raise ValueError(f"Invalid sex values: {sorted(bad_sex)}")
    if not network_columns(df):
        raise ValueError("CSV has no network columns (expected net_<Yeo7Name> or net_1..net_7)")
    return df


def write_m0_report(report: M0Report, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def load_m0_report(path: Path) -> M0Report:
    return M0Report.model_validate(json.loads(path.read_text(encoding="utf-8")))


def m0_passed_for_training(path: Path) -> bool:
    """Return True only if a real-data, multi-site M0 report passed all go criteria."""
    if not path.is_file():
        return False
    report = load_m0_report(path)
    if report.is_synthetic:
        return False
    if not report.passed:
        return False
    if report.n_sites < 2 and not report.multi_site_validated:
        return False
    if "not_production_gate" in report.notes or "hcp_sex_only_probe" in report.notes:
        return False
    return True
