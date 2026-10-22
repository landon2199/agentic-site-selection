"""Tool implementations for the site-selection MCP server (plain functions that return JSON-able dicts).

Every result carries:
  sources  where each number came from (dataset, run, engine)
  checks   sanity checks with status pass / warn / fail
  caveats  what the numbers cannot tell you

Data tiers (the server degrades instead of failing):
  1. results/hex_scores.csv     committed derived table: open ACS/shop fields + both methods' scores. Always there.
  2. data/esri/embeddings_local.parquet  local copy of Esri's embeddings (git-ignored; needs an ArcGIS org
                                sign-in to create). Enables custom find_similar queries in numpy.
  3. ArcGIS Pro 3.7 Advanced    enables find_similar(engine="arcpy"), which calls
                                arcpy.geoai.FindSimilarFeaturesUsingEmbeddings in a subprocess.
Without tiers 2-3, find_similar falls back to an open proxy (cosine on z-scored ACS + shop features) and says so.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import urllib.parse
from functools import lru_cache
from pathlib import Path

import h3
import httpx
import numpy as np
import pandas as pd

from sitesel import core

ROOT = core.ROOT
ARCGIS_PY = Path("C:/Program Files/ArcGIS/Pro/bin/Python/envs/arcgispro-py3/python.exe")
GEOCODER = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
H3_RES = 7
MAX_TOP_N = 50
HOT_SPOT_FIELDS = ["overlay", "emb", "pop", "pop_density_km2", "share_25_34", "med_hh_income", "n_shops",
                   "n_indep", "n_chain", "dist_nearest_shop_km"]
OPEN_PROXY_FEATURES = ["share_25_34", "med_hh_income", "pop_density_km2", "n_shops_ring1", "dist_nearest_shop_km"]

SRC_ACS = "ACS 2020-2024 5-year (B01001, B01003, B19013), tract -> hex by 2020 block-population weights"
SRC_SHOPS = "Overture Maps Places release 2026-10-21.0 (coffee_shop / cafe / roastery), cleaned and de-duplicated"
SRC_EMB = ("Esri USA Geodemographic Embeddings 2026 (beta, Living Atlas 6637a530b840446285a4383627a13f1f), "
           "H3 r7, copied to a local file geodatabase")

CAVEATS_COMMON = [
    "No sales or revenue data: 'like existing shops' is not the same as 'profitable'.",
    "H3 r7 hexes are about 5.2 km^2, coarse for a coffee shop whose catchment is often a few blocks.",
    "Shop locations come from Overture Places; openings, closures and independents with no web presence can be missing.",
]


class ToolInputError(ValueError):
    """Bad input; the message is safe to show to the assistant."""


# ------------------------------------------------------------------ data access
@lru_cache(maxsize=1)
def scores() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "hex_scores.csv").set_index("h3")


@lru_cache(maxsize=1)
def summary() -> dict:
    return json.loads((ROOT / "results" / "summary.json").read_text())


@lru_cache(maxsize=1)
def shops() -> pd.DataFrame | None:
    p = core.DATA / "shops.parquet"
    return pd.read_parquet(p, columns=["name", "is_chain", "h3", "lon", "lat"]) if p.exists() else None


@lru_cache(maxsize=1)
def embeddings() -> pd.DataFrame | None:
    return core.load_embeddings()


def capabilities() -> dict:
    return {"derived_scores": True, "local_embeddings": embeddings() is not None,
            "arcgis_pro": ARCGIS_PY.exists() and (core.ESRI / "embeddings.gdb").exists(),
            "shop_names": shops() is not None}


def _check(name: str, ok: bool, detail: str, warn_only: bool = False) -> dict:
    return {"check": name, "status": "pass" if ok else ("warn" if warn_only else "fail"), "detail": detail}


def _hex_id(h3_id: str | None, lon: float | None, lat: float | None) -> str:
    if h3_id:
        if not h3.is_valid_cell(h3_id) or h3.get_resolution(h3_id) != H3_RES:
            raise ToolInputError(f"{h3_id!r} is not a valid H3 resolution-7 cell id.")
        return h3_id
    if lon is None or lat is None:
        raise ToolInputError("Give h3_id, or both lon and lat.")
    if 25 < lon < 35 and -100 < lat < -90:
        raise ToolInputError(f"lon={lon}, lat={lat} looks swapped (Houston is about lon -95.4, lat 29.8).")
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ToolInputError("lon/lat out of range (lon -180..180, lat -90..90).")
    return h3.latlng_to_cell(lat, lon, H3_RES)


def _row(h: str, s: pd.DataFrame, rank_col: str | None = None) -> dict:
    r = s.loc[h]
    out = {"h3": h, "lon": round(float(r.lon), 5), "lat": round(float(r.lat), 5),
           "population": int(round(r["pop"])), "share_25_34": round(float(r.share_25_34), 3) if pd.notna(r.share_25_34) else None,
           "median_hh_income_usd": int(r.med_hh_income) if pd.notna(r.med_hh_income) else None,
           "pop_density_per_km2": round(float(r.pop_density_km2)) if pd.notna(r.pop_density_km2) else None,
           "coffee_shops": int(r.n_shops), "independents": int(r.n_indep), "chains": int(r.n_chain),
           "shops_in_hex_and_neighbours": int(r.n_shops_ring1),
           "km_to_nearest_shop_from_centre": round(float(r.dist_nearest_shop_km), 2)}
    return out


# ------------------------------------------------------------------ geocode
def geocode(address: str) -> dict:
    address = (address or "").strip()
    if not 5 <= len(address) <= 200:
        raise ToolInputError("Give a one-line U.S. street address (5-200 characters).")
    url = f"{GEOCODER}?{urllib.parse.urlencode({'address': address, 'benchmark': 'Public_AR_Current', 'format': 'json'})}"
    try:
        r = httpx.get(url, timeout=30)
        r.raise_for_status()
        matches = r.json()["result"]["addressMatches"]
    except Exception as exc:  # network / service errors become a readable message
        raise ToolInputError(f"Census geocoder unavailable ({type(exc).__name__}); try again or give lon/lat.") from None
    if not matches:
        return {"matched": False, "address": address, "sources": {"geocoder": "U.S. Census Geocoder, Public_AR_Current"},
                "checks": [_check("address_matched", False, "no match; check spelling or add city/ZIP")],
                "caveats": ["The Census geocoder matches street address ranges only (no business names or landmarks)."]}
    m = matches[0]
    lon, lat = m["coordinates"]["x"], m["coordinates"]["y"]
    cell = h3.latlng_to_cell(lat, lon, H3_RES)
    inside = cell in scores().index
    return {"matched": True, "matched_address": m["matchedAddress"], "lon": round(lon, 6), "lat": round(lat, 6),
            "h3": cell, "in_study_area": inside, "n_candidates": len(matches),
            "sources": {"geocoder": "U.S. Census Geocoder, benchmark Public_AR_Current (free, no ArcGIS credits)"},
            "checks": [_check("address_matched", True, m["matchedAddress"]),
                       _check("in_study_area", inside, "hex is in the Harris County study area" if inside
                              else "outside the study area (Harris County, TX); no scores exist there")],
            "caveats": ["Address-range interpolation: position can be off by tens of metres.",
                        *(["More than one candidate matched; the first was used."] if len(matches) > 1 else [])]}


# ------------------------------------------------------------------ hex_profile
def hex_profile(h3_id: str | None = None, lon: float | None = None, lat: float | None = None) -> dict:
    h = _hex_id(h3_id, lon, lat)
    s = scores()
    if h not in s.index:
        raise ToolInputError(f"Hex {h} is outside the study area (Harris County, TX).")
    r = s.loc[h]
    prof = _row(h, s)
    prof["overlay_score_1_10"] = round(float(r.overlay), 2)
    prof["overlay_percentile"] = round(float(r.overlay_pct) * 100, 1)
    prof["overlay_masked"] = bool(r.overlay_masked)
    prof["overlay_subscores_1_10"] = {k[2:]: round(float(r[k]), 2) for k in s.columns if k.startswith("r_")}
    prof["embedding_similarity"] = round(float(r.emb), 4)
    prof["embedding_percentile"] = round(float(r.emb_pct) * 100, 1)
    prof["agreement_top10"] = r.agreement
    prof["overlay_hot_spot_gi_z"] = round(float(r.overlay_gi_z), 2)
    sh = shops()
    names = None
    if sh is not None:
        here = sh[sh.h3 == h]
        names = {"independent": sorted(here.name[~here.is_chain].tolist())[:12],
                 "chain": sorted(here.name[here.is_chain].tolist())[:12]}
    checks = [_check("population_present", r["pop"] > 0, f"{r['pop']:.0f} residents allocated"),
              _check("mostly_in_study_area", r.frac_in_area >= 0.5, f"{r.frac_in_area:.0%} of the hex is inside the county",
                     warn_only=True),
              _check("not_masked", not r.overlay_masked, "overlay mask: pop >= 500 and land >= 1 km^2", warn_only=True)]
    return {"hex": prof, "shop_names": names,
            "sources": {"acs": SRC_ACS, "shops": SRC_SHOPS, "embedding_scores": summary().get("embedding_source"),
                        "embeddings": SRC_EMB, "table": "results/hex_scores.csv"},
            "checks": checks,
            "caveats": CAVEATS_COMMON + ["Median household income for a hex is a population-weighted mean of tract "
                                         "medians, not a true median."]}


# ------------------------------------------------------------------ overlay
def overlay_suitability(weights: dict | None = None, top_n: int = 10, exclude_existing_independents: bool = False) -> dict:
    top_n = int(top_n)
    if not 1 <= top_n <= MAX_TOP_N:
        raise ToolInputError(f"top_n must be 1-{MAX_TOP_N}.")
    s = scores().reset_index()
    try:
        ov = core.overlay(s, weights=weights)
    except ValueError as exc:
        raise ToolInputError(str(exc)) from None
    s["score"] = ov.overlay.values
    base = s.overlay
    rho = core.spearman(s.set_index("h3").score, s.set_index("h3").overlay)
    pool = s[~ov.masked.values]
    if exclude_existing_independents:
        pool = pool[pool.n_indep == 0]
    top = pool.sort_values("score", ascending=False).head(top_n)
    sc = scores()
    ranked = []
    for i, (_, r) in enumerate(top.iterrows(), 1):
        d = _row(r.h3, sc)
        d.update({"rank": i, "score_1_10": round(float(r.score), 2),
                  "subscores_1_10": {k[2:]: round(float(ov.loc[_, k]), 2) for k in ov.columns if k.startswith("r_")},
                  "embedding_percentile": round(float(r.emb_pct) * 100, 1)})
        ranked.append(d)
    w = ov.attrs["weights"]
    cfg = core.load_yaml("overlay.yaml")
    checks = [_check("weights_sum_to_1", abs(sum(w.values()) - 1) < 1e-9, json.dumps({k: round(v, 3) for k, v in w.items()})),
              _check("scores_in_range", bool(ov.overlay.between(1, 10).all()), "all scores within 1-10"),
              _check("rank_stability_vs_default", rho >= 0.8, f"Spearman vs default weights = {rho:.3f}", warn_only=True)]
    return {"weights_used": {k: round(v, 3) for k, v in w.items()},
            "criteria": {k: {"function": c["function"], "direction": c["direction"],
                             "range_percentiles": [c["low_pct"], c["high_pct"]], "why": c["why"]}
                         for k, c in cfg["criteria"].items()},
            "masked_hexes": int(ov.masked.sum()), "n_hexes": len(s), "top": ranked,
            "sources": {"method": "Weighted overlay (rescale to 1-10, weighted sum), config/overlay.yaml",
                        "acs": SRC_ACS, "shops": SRC_SHOPS},
            "checks": checks,
            "caveats": CAVEATS_COMMON + ["Weights are expert judgement, not fitted to outcomes.",
                                         "The competition criterion rewards gaps; a hex with zero shops may also have zero demand."]}


# ------------------------------------------------------------------ find_similar
def _open_proxy(query: list[str]) -> pd.Series:
    s = scores()
    X = s[OPEN_PROXY_FEATURES].copy()
    X["med_hh_income"] = np.log(X.med_hh_income)
    X["pop_density_km2"] = np.log1p(X.pop_density_km2)
    X = (X - X.mean()) / X.std()
    X = X.fillna(0)
    return core.similar_numpy(X.astype(float), query).rename("score")


def _arcpy_similar(query: list[str]) -> pd.Series:
    with tempfile.TemporaryDirectory() as tmp:
        q = Path(tmp) / "query.txt"
        q.write_text("\n".join(query))
        name = "mcp_query"
        r = subprocess.run([str(ARCGIS_PY), "-u", str(ROOT / "pipeline" / "arcpy_tools.py"), "similar", str(q), name],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise ToolInputError(f"ArcPy run failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'unknown'}")
    return core.esri_scores(name).rename("score")


def find_similar(query: list[str] | str = "independent_shops", top_n: int = 10, exclude_existing: bool = True,
                 engine: str = "auto") -> dict:
    """See server docstring."""
    top_n = int(top_n)
    if not 1 <= top_n <= MAX_TOP_N:
        raise ToolInputError(f"top_n must be 1-{MAX_TOP_N}.")
    if engine not in ("auto", "arcpy", "numpy", "open_proxy"):
        raise ToolInputError("engine must be auto, arcpy, numpy or open_proxy.")
    s = scores()
    default = query == "independent_shops"
    if default:
        q = s.index[s.n_indep > 0].tolist()
    else:
        if isinstance(query, str):
            query = [query]
        bad = [x for x in query if not (isinstance(x, str) and h3.is_valid_cell(x) and h3.get_resolution(x) == H3_RES)]
        if bad:
            raise ToolInputError(f"query must be 'independent_shops' or a list of H3 r7 ids; invalid: {bad[:3]}")
        q = [x for x in dict.fromkeys(query) if x in s.index]
        if not q:
            raise ToolInputError("none of the query hexes is in the study area.")
        if len(q) > 400:
            raise ToolInputError("at most 400 query hexes.")
    cap = capabilities()
    used = engine
    if engine == "auto":
        used = "precomputed_esri" if default and "emb_esri" in s else ("numpy" if cap["local_embeddings"] else "open_proxy")
    if used == "arcpy" and not cap["arcgis_pro"]:
        raise ToolInputError("ArcGIS Pro / local embeddings GDB not available here; use engine='auto'.")
    if used == "numpy" and not cap["local_embeddings"]:
        raise ToolInputError("Local embeddings not available (needs an ArcGIS org sign-in to export); use engine='auto'.")
    if used == "precomputed_esri":
        sc = s.emb_esri.rename("score")
    elif used == "arcpy":
        sc = _arcpy_similar(q)
    elif used == "numpy":
        sc = core.similar_numpy(embeddings(), q).rename("score")
    else:
        sc = _open_proxy(q)
    sc = sc.reindex(s.index)
    pct = sc.rank(pct=True)
    checks = [_check("all_hexes_scored", bool(sc.notna().all()), f"{int(sc.notna().sum())} of {len(s)} hexes scored"),
              _check("query_hexes_rank_high", float(pct[q].median()) >= 0.6,
                     f"median percentile of the {len(q)} query hexes = {pct[q].median() * 100:.0f} (self-consistency)",
                     warn_only=True)]
    if used in ("numpy", "arcpy") and default and "emb_esri" in s:
        rho = core.spearman(sc, s.emb_esri)
        checks.append(_check("matches_esri_tool", rho > 0.999, f"Spearman vs Esri precomputed run = {rho:.6f}"))
    if used == "precomputed_esri" and "esri_vs_numpy" in summary():
        e = summary()["esri_vs_numpy"]
        checks.append(_check("esri_vs_numpy_fallback", e["spearman"] > 0.999,
                             f"numpy cosine re-implementation Spearman {e['spearman']}, max |diff| {e['max_abs_diff']:.1e}"))
    pool = s.assign(score=sc)
    if exclude_existing:
        pool = pool[pool.n_indep == 0]
    pool = pool[~pool.index.isin(q)].sort_values("score", ascending=False).head(top_n)
    ranked = []
    for i, (h, r) in enumerate(pool.iterrows(), 1):
        d = _row(h, s)
        d.update({"rank": i, "similarity": round(float(r.score), 4), "similarity_percentile": round(float(pct[h]) * 100, 1),
                  "overlay_score_1_10": round(float(r.overlay), 2)})
        ranked.append(d)
    engines = {"precomputed_esri": "arcpy.geoai.FindSimilarFeaturesUsingEmbeddings (ArcGIS Pro 3.7), precomputed run",
               "arcpy": "arcpy.geoai.FindSimilarFeaturesUsingEmbeddings (ArcGIS Pro 3.7), live subprocess",
               "numpy": "numpy cosine similarity to the mean query embedding (identical ranking to the Esri tool)",
               "open_proxy": "OPEN PROXY, not the Esri embeddings: cosine on z-scored ACS + shop features"}
    cav = CAVEATS_COMMON + ["Similarity means 'demographically like the query hexes', not 'will succeed'.",
                            "The query is averaged into one vector, so a few very different query hexes blur together."]
    if used == "open_proxy":
        cav.insert(0, "Esri embeddings were not available, so an open proxy was used; results are NOT the embedding method.")
    else:
        cav.append("USA Geodemographic Embeddings 2026 are a beta dataset; their inputs may include the same ACS fields used by the overlay.")
    return {"engine": used, "engine_detail": engines[used], "n_query_hexes": len(q),
            "query": "hexes with >=1 independent coffee shop" if default else q[:20],
            "excluded": ("query hexes" + (" and hexes that already have an independent shop" if exclude_existing else "")),
            "top": ranked,
            "sources": {"embeddings": SRC_EMB if used != "open_proxy" else None, "acs": SRC_ACS, "shops": SRC_SHOPS},
            "checks": checks, "caveats": cav}


# ------------------------------------------------------------------ hot spots
def hot_spots(field: str = "overlay", top_n: int = 10) -> dict:
    if field not in HOT_SPOT_FIELDS:
        raise ToolInputError(f"field must be one of {HOT_SPOT_FIELDS}")
    top_n = int(top_n)
    if not 1 <= top_n <= MAX_TOP_N:
        raise ToolInputError(f"top_n must be 1-{MAX_TOP_N}.")
    s = scores()
    vals = s[field].fillna(s[field].median())
    gi = core.gi_star(s.index.tolist(), vals.to_numpy()).set_index("h3")
    hot = gi[(gi.gi_z >= 1.96)]
    cold = gi[(gi.gi_z <= -1.96)]
    top = gi.sort_values("gi_z", ascending=False).head(top_n)
    rows = [{**_row(h, s), "gi_z": round(float(r.gi_z), 2), "p": round(float(r.gi_p), 4), field: round(float(s.loc[h, field]), 4)}
            for h, r in top.iterrows()]
    checks = [_check("nonconstant_field", float(vals.std()) > 0, f"std of {field} = {vals.std():.4g}"),
              _check("filled_missing", int(s[field].isna().sum()) == 0, f"{int(s[field].isna().sum())} missing values filled with median",
                     warn_only=True)]
    if field == "overlay" and "gi_numpy_vs_arcpy" in summary():
        g = summary()["gi_numpy_vs_arcpy"]
        checks.append(_check("matches_arcpy_hotspots", g["spearman"] > 0.999,
                             f"arcpy.stats.HotSpots (contiguity edges+corners): Spearman {g['spearman']}, max |dz| {g['max_abs_z_diff']}"))
    return {"field": field, "n_hot_95": int(len(hot)), "n_cold_95": int(len(cold)), "n_hexes": len(s), "hottest": rows,
            "sources": {"method": "Getis-Ord Gi*, binary weights = hex + 6 H3 neighbours, numpy (sitesel.core.gi_star)",
                        "table": "results/hex_scores.csv"},
            "checks": checks,
            "caveats": ["No multiple-testing correction (FDR); with ~880 hexes about 44 would pass p<0.05 by chance.",
                        "Hot spots of a score show where high scores cluster, not where any single site is best."]}


# ------------------------------------------------------------------ validate
def validate(method: str = "all") -> dict:
    v = json.loads((ROOT / "results" / "validation.json").read_text())
    blocks = pd.read_csv(ROOT / "results" / "validation_blocks.csv")
    alias = {"embeddings": "embeddings_esri", "overlay": "overlay", "overlay_demand_only": "overlay_demand_only",
             "embeddings_esri": "embeddings_esri", "embeddings_numpy": "embeddings_numpy", "random": "random"}
    if method != "all" and method not in alias:
        raise ToolInputError(f"method must be 'all' or one of {sorted(alias)}")
    keep = list(v["per_method"]) if method == "all" else [alias[method], "random"]
    per = {m: v["per_method"][m] for m in keep if m in v["per_method"]}
    tbl = blocks[blocks.method.isin(keep)][["block", "method", "n_hexes", "n_pos", "auc", "p_at_k", "p_at_10"]].round(3)
    rnd = v["per_method"]["random"]
    checks = []
    for m, d in per.items():
        if m == "random":
            continue
        beats = d["auc"]["mean"] > 0.5 and d["p_at_k"]["mean"] > rnd["p_at_k"]["mean"]
        checks.append(_check(f"{m}_beats_random", beats, f"AUC {d['auc']['mean']} (range {d['auc']['min']}-{d['auc']['max']}), "
                             f"P@k {d['p_at_k']['mean']} vs random {rnd['p_at_k']['mean']}", warn_only=True))
        if d["auc"]["min"] < 0.6:
            checks.append(_check(f"{m}_every_block_useful", False, f"weakest block AUC {d['auc']['min']}", warn_only=True))
    return {"design": v["design"], "per_method_mean_min_max": per, "per_block": tbl.to_dict("records"),
            "method_agreement": summary().get("method_agreement"),
            "sources": {"script": "pipeline/validate.py", "files": ["results/validation.json", "results/validation_blocks.csv"]},
            "checks": checks,
            "caveats": ["Positives are hexes that already have an independent shop: this measures 'finds places like where "
                        "shops are', not profitability or survival.",
                        "Only 5 blocks; the min-max range is wide and one block (the dense urban core) behaves differently.",
                        "Within a held-out block the competition criterion is nearly constant (its shops are hidden), so "
                        "overlay and overlay_demand_only score almost the same there."]}
