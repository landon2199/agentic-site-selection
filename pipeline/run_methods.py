"""Score every hex with both methods, measure agreement, precompute hot spots, and write the derived outputs.

    python pipeline/run_methods.py

Writes (all derived, safe to commit; no embedding values):
  results/hex_scores.csv     per hex: open ACS/shop fields, overlay score, embedding similarity (Esri tool and
                             numpy), percentile ranks, top-10% flags, agreement class, Gi* of overlay
  results/summary.json       Esri-vs-numpy agreement, method agreement (Jaccard top 10%, Spearman), counts
  docs/data/hexes.geojson    the same table with hex polygons, for the static web page
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sitesel import core  # noqa: E402

ARCGIS_PY = Path("C:/Program Files/ArcGIS/Pro/bin/Python/envs/arcgispro-py3/python.exe")
OPEN_COLS = ["h3", "lon", "lat", "frac_in_area", "pop", "share_25_34", "med_hh_income", "pop_density_km2",
             "land_km2", "n_shops", "n_indep", "n_chain", "n_shops_ring1", "dist_nearest_shop_km"]


def main() -> None:
    h = core.load_hexes().reset_index(drop=True)
    emb = core.load_embeddings()
    query = h.h3[h.n_indep > 0].tolist()
    ov = core.overlay(h)
    out = h[OPEN_COLS].copy()
    out["overlay"] = ov.overlay.values
    out["overlay_masked"] = ov.masked.values
    for c in [c for c in ov.columns if c.startswith("r_")]:
        out[c] = ov[c].values
    summary: dict = {"n_hexes": len(h), "n_query_hexes": len(query), "overlay_weights": ov.attrs["weights"]}

    if emb is not None:
        out["emb_numpy"] = core.similar_numpy(emb, query).reindex(out.h3).values
    esri = core.esri_scores("all")
    if esri is not None:
        out["emb_esri"] = esri.reindex(out.h3).values
        if "emb_numpy" in out:
            a, b = out.set_index("h3").emb_esri, out.set_index("h3").emb_numpy
            summary["esri_vs_numpy"] = {"spearman": round(core.spearman(a, b), 6),
                                        "max_abs_diff": float((a - b).abs().max()),
                                        "top10_jaccard": core.jaccard(core.top_share(a), core.top_share(b))}
    emb_col = "emb_esri" if "emb_esri" in out else ("emb_numpy" if "emb_numpy" in out else None)
    out["emb"] = out[emb_col] if emb_col else np.nan
    summary["embedding_source"] = emb_col

    # Percentile ranks; agreement uses unmasked hexes so empty land does not count as "agreement"
    for c in ("overlay", "emb"):
        out[f"{c}_pct"] = out[c].rank(pct=True)
    s = out.set_index("h3")
    t_ov, t_em = core.top_share(s.overlay), core.top_share(s.emb)
    out["top10_overlay"] = out.h3.isin(t_ov)
    out["top10_emb"] = out.h3.isin(t_em)
    out["agreement"] = np.select([out.top10_overlay & out.top10_emb, out.top10_overlay, out.top10_emb],
                                 ["both", "overlay only", "embeddings only"], "neither")
    summary["method_agreement"] = {
        "top10_jaccard": round(core.jaccard(t_ov, t_em), 3), "top10_both": len(t_ov & t_em),
        "top10_size": len(t_ov), "spearman_all": round(core.spearman(s.overlay, s.emb), 3),
        "spearman_unmasked": round(core.spearman(s.overlay[~s.overlay_masked], s.emb[~s.overlay_masked]), 3)}

    # Hot spots of overlay suitability: numpy Gi* (always) + ArcPy HotSpots cross-check (if available)
    gi = core.gi_star(out.h3.tolist(), out.overlay.to_numpy())
    out["overlay_gi_z"] = gi.gi_z.values
    if ARCGIS_PY.exists() and (core.ESRI / "embeddings.gdb").exists():
        vals = core.ESRI / "overlay_values.csv"
        out[["h3", "overlay"]].rename(columns={"overlay": "value"}).to_csv(vals, index=False)
        r = subprocess.run([str(ARCGIS_PY), "-u", str(core.ROOT / "pipeline" / "arcpy_tools.py"), "hotspots",
                            str(vals), "hotspots_overlay"], capture_output=True, text=True, timeout=1800)
        print(r.stdout[-400:], r.stderr[-400:])
        p = core.ESRI / "hotspots_overlay.csv"
        if p.exists():
            a = pd.read_csv(p).set_index("h3").gi_z
            summary["gi_numpy_vs_arcpy"] = {"spearman": round(core.spearman(a, out.set_index("h3").overlay_gi_z), 4),
                                            "max_abs_z_diff": round(float((a - out.set_index("h3").overlay_gi_z).abs().max()), 4)}

    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    core.RESULTS.mkdir(exist_ok=True)
    out.to_csv(core.RESULTS / "hex_scores.csv", index=False)
    (core.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    g = gpd.read_parquet(core.DATA / "hexes_open.parquet")[["h3", "geometry"]].merge(out, on="h3")
    g = g.drop(columns=[c for c in g.columns if c.startswith("r_")] + ["emb_numpy"], errors="ignore")
    docs = core.ROOT / "docs" / "data"
    docs.mkdir(parents=True, exist_ok=True)
    g.to_file(docs / "hexes.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
