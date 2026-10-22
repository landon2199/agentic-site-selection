"""Core scoring used by the pipeline, the MCP server and the validation (pure numpy / pandas; no ArcPy).

Method 1  overlay():      rescale criteria to 1-10 (config/overlay.yaml), weighted sum, mask.
Method 2  similar_numpy(): cosine similarity of every hex embedding to the query embedding. This is the open
                           re-implementation of arcpy.geoai.FindSimilarFeaturesUsingEmbeddings; the Esri tool's
                           own scores are read from data/esri/*.csv when present.
"""

from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA, ESRI, RESULTS = ROOT / "data", ROOT / "data" / "esri", ROOT / "results"


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ data
def load_hexes() -> pd.DataFrame:
    """Open hex table (no embeddings). GeoParquet read as a plain frame; geometry kept as WKB."""
    return pd.read_parquet(DATA / "hexes_open.parquet")


def load_shops() -> pd.DataFrame:
    return pd.read_parquet(DATA / "shops.parquet")


def load_embeddings() -> pd.DataFrame | None:
    """h3 x 256 float32 Esri geodemographic embeddings (local, git-ignored). None if not exported."""
    p = ESRI / "embeddings_local.parquet"
    return pd.read_parquet(p).set_index("h3") if p.exists() else None


# ------------------------------------------------------------------ method 1
def rescale(x: pd.Series, function: str, direction: str, low_pct: float, high_pct: float) -> pd.Series:
    """Rescale to 1-10. 'linear' or 'log_linear' between two percentiles, clipped; 'decreasing' flips it."""
    v = x.astype(float)
    if function == "log_linear":
        v = np.log1p(v.clip(lower=0))
    elif function != "linear":
        raise ValueError(f"unknown function {function}")
    lo, hi = np.nanpercentile(v, low_pct), np.nanpercentile(v, high_pct)
    t = ((v - lo) / (hi - lo if hi > lo else 1)).clip(0, 1)
    if direction == "decreasing":
        t = 1 - t
    return 1 + 9 * t


def competition(hexes: pd.DataFrame, shops: pd.DataFrame) -> pd.Series:
    """Coffee shops in each hex plus its 6 neighbours, recomputed from a (possibly reduced) shop table."""
    per = shops.groupby("h3").size()
    return pd.Series([int(sum(per.get(n, 0) for n in h3.grid_disk(h, 1))) for h in hexes.h3], index=hexes.index)


def overlay(hexes: pd.DataFrame, weights: dict | None = None, shops: pd.DataFrame | None = None,
            cfg: dict | None = None) -> pd.DataFrame:
    """Weighted-overlay suitability. `weights` overrides config weights (renormalised to sum 1).
    `shops` (optional) recomputes the competition criterion, e.g. with held-out shops removed."""
    cfg = cfg or load_yaml("overlay.yaml")
    crit = cfg["criteria"]
    w = {k: float(c["weight"]) for k, c in crit.items()}
    if weights:
        unknown = set(weights) - set(w)
        if unknown:
            raise ValueError(f"unknown criteria {sorted(unknown)}; valid: {sorted(w)}")
        w.update({k: float(v) for k, v in weights.items()})
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("weights must be >= 0 and not all zero")
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}
    h = hexes.copy()
    if shops is not None:
        h["n_shops_ring1"] = competition(h, shops)
    out = pd.DataFrame(index=h.index)
    out["h3"] = h.h3
    score = 0
    for k, c in crit.items():
        r = rescale(h[k], c["function"], c["direction"], c["low_pct"], c["high_pct"]).fillna(1.0)
        out[f"r_{k}"] = r
        score = score + w[k] * r
    m = cfg["mask"]
    masked = (h["pop"] < m["min_pop"]) | (h["land_km2"] < m["min_land_km2"])
    out["masked"] = masked
    out["overlay"] = np.where(masked, 1.0, score)
    out.attrs["weights"] = w
    return out


# ------------------------------------------------------------------ method 2
def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(n == 0, 1, n)


def similar_numpy(emb: pd.DataFrame, query_ids: list[str], how: str = "mean") -> pd.Series:
    """Cosine similarity of every hex to the query. how='mean': the mean of the query vectors (one query
    embedding, as the Esri tool reports 'Building query embedding'); how='unit_mean': mean of unit vectors;
    how='max': best match to any single query hex."""
    q = [i for i in query_ids if i in emb.index]
    if not q:
        raise ValueError("none of the query hexes has an embedding")
    X = emb.to_numpy(np.float64)
    Q = emb.loc[q].to_numpy(np.float64)
    if how == "mean":
        s = _unit(X) @ _unit(Q.mean(0, keepdims=True)).T
        s = s[:, 0]
    elif how == "unit_mean":
        s = (_unit(X) @ _unit(_unit(Q).mean(0, keepdims=True)).T)[:, 0]
    elif how == "max":
        s = (_unit(X) @ _unit(Q).T).max(1)
    else:
        raise ValueError(how)
    return pd.Series(s, index=emb.index, name="emb_numpy")


def esri_scores(name: str = "all") -> pd.Series | None:
    p = ESRI / f"{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d.groupby("h3").esri_score.max()


# ------------------------------------------------------------------ hot spots
def gi_star(h3_ids: list[str], values: np.ndarray) -> pd.DataFrame:
    """Getis-Ord Gi* with binary H3 ring-1 weights including the hex itself (the hex and its 6 neighbours;
    hexes outside the study area are absent). Returns z-scores and two-sided normal p-values."""
    from math import erf, sqrt

    x = np.asarray(values, float)
    idx = {h: i for i, h in enumerate(h3_ids)}
    n, xbar = len(x), x.mean()
    s = np.sqrt((x ** 2).mean() - xbar ** 2)
    z = np.empty(n)
    for i, h in enumerate(h3_ids):
        nb = [idx[g] for g in h3.grid_disk(h, 1) if g in idx]
        w = len(nb)
        num = x[nb].sum() - xbar * w
        den = s * np.sqrt((n * w - w ** 2) / (n - 1))
        z[i] = num / den if den > 0 else 0.0
    p = np.array([2 * (1 - 0.5 * (1 + erf(abs(v) / sqrt(2)))) for v in z])
    return pd.DataFrame({"h3": h3_ids, "gi_z": z, "gi_p": p})


# ------------------------------------------------------------------ metrics
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-np.asarray(scores, float), kind="stable")[:k]
    return float(np.asarray(labels, bool)[order].mean()) if k > 0 else float("nan")


def top_share(scores: pd.Series, share: float = 0.10) -> set:
    k = max(1, int(round(len(scores) * share)))
    return set(scores.sort_values(ascending=False).index[:k])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


def spearman(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    return float(j.iloc[:, 0].rank().corr(j.iloc[:, 1].rank()))
