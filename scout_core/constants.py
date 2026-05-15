"""Canonical Yeo-7-style labels aligned with scripts/build_vertex_regions_csv demo."""

YEO7_NAMES = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)

NETWORK_ID_TO_NAME: dict[int, str] = {i + 1: name for i, name in enumerate(YEO7_NAMES)}
NETWORK_NAME_TO_ID: dict[str, int] = {name: i + 1 for i, name in enumerate(YEO7_NAMES)}


def network_names_for_ids(net_ids: list[int]) -> list[str]:
    return [NETWORK_ID_TO_NAME.get(int(i), f"net_{i}") for i in net_ids]
