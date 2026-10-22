"""Spatially blocked hold-out validation of both methods.

    python pipeline/validate.py            # numpy methods + the Esri tool if ArcGIS Pro is installed
    python pipeline/validate.py --no-esri  # open fallback only

Design
  * Positives: hexes that hold at least one independent coffee shop (Overture Places, cleaned).
  * Blocks: k-means (k=5, seed 0) on hex centroids in CONUS Albers, so held-out hexes are spatially apart
    from most of the query hexes.
  * For each block: hide every shop in the block, rebuild the query from the independent-shop hexes in the
    other four blocks, score all hexes, and evaluate inside the block only:
      AUC        shop hexes vs non-shop hexes (0.5 = random)
      P@k        share of shop hexes among the block's top k, k = number of held-out shop hexes (R-precision)
      P@10       share of shop hexes among the block's top 10
    Random baseline: 2,000 random orderings per block.
  * Methods: overlay (config weights, competition recomputed without the hidden shops), overlay demand-only
    (competition weight 0), embeddings (numpy cosine, identical to the Esri tool), embeddings via the Esri tool.
Writes results/validation.json and results/validation_blocks.csv.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sitesel import core  # noqa: E402

ARCGIS_PY = Path("C:/Program Files/ArcGIS/Pro/bin/Python/envs/arcgispro-py3/python.exe")
K_BLOCKS, SEED, N_RANDOM = 5, 0, 2000


def blocks(h: pd.DataFrame) -> np.ndarray:
    import geopandas as gpd

    p = gpd.GeoSeries(gpd.points_from_xy(h.lon, h.lat), crs=4326).to_crs("EPSG:5070")
    return KMeans(K_BLOCKS, random_state=SEED, n_init=10).fit_predict(np.c_[p.x, p.y])


def main(use_esri: bool = True) -> dict:
    h = core.load_hexes().reset_index(drop=True)
    shops = core.load_shops()
    emb = core.load_embeddings()
    h["block"] = blocks(h)
    h["pos"] = h.n_indep > 0
    pos_ids = set(h.h3[h.pos])

    # Esri tool: one ArcPy session for all 5 query sets
    esri_ok = use_esri and emb is not None and ARCGIS_PY.exists()
    if esri_ok:
        args = []
        for b in range(K_BLOCKS):
            q = h.h3[h.pos & (h.block != b)]
            f = core.ESRI / f"query_block{b}.txt"
            f.write_text("\n".join(q))
            args += [str(f), f"block{b}"]
        r = subprocess.run([str(ARCGIS_PY), "-u", str(core.ROOT / "pipeline" / "arcpy_tools.py"), "similar", *args],
                           capture_output=True, text=True, timeout=1800)
        print(r.stdout[-800:], r.stderr[-800:])
        esri_ok = r.returncode == 0

    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(K_BLOCKS):
        inb = h.block == b
        hidden = set(h.h3[inb])
        visible = shops[~shops.h3.isin(hidden)]
        query = sorted(pos_ids - hidden)
        y = h.pos[inb].to_numpy()
        n_pos, k10 = int(y.sum()), 10
        methods = {
            "overlay": core.overlay(h, shops=visible).overlay,
            "overlay_demand_only": core.overlay(h, weights={"n_shops_ring1": 0}, shops=visible).overlay,
        }
        if emb is not None:
            methods["embeddings_numpy"] = core.similar_numpy(emb, query).reindex(h.h3).reset_index(drop=True)
        if esri_ok:
            methods["embeddings_esri"] = core.esri_scores(f"block{b}").reindex(h.h3).reset_index(drop=True)
        for name, s in methods.items():
            sc = s[inb].to_numpy()
            rows.append({"block": b, "method": name, "n_hexes": int(inb.sum()), "n_pos": n_pos,
                         "n_query": len(query), "auc": core.auc(sc, y),
                         "p_at_k": core.precision_at_k(sc, y, n_pos), "p_at_10": core.precision_at_k(sc, y, k10)})
        rnd = [(core.precision_at_k(r, y, n_pos), core.precision_at_k(r, y, k10), core.auc(r, y))
               for r in rng.random((N_RANDOM, int(inb.sum())))]
        rnd = np.array(rnd)
        rows.append({"block": b, "method": "random", "n_hexes": int(inb.sum()), "n_pos": n_pos, "n_query": len(query),
                     "auc": float(rnd[:, 2].mean()), "p_at_k": float(rnd[:, 0].mean()), "p_at_10": float(rnd[:, 1].mean()),
                     "p_at_k_random_95": float(np.percentile(rnd[:, 0], 95))})
    df = pd.DataFrame(rows)
    df.to_csv(core.RESULTS / "validation_blocks.csv", index=False)
    summ = {}
    for m, g in df.groupby("method", sort=False):
        summ[m] = {met: {"mean": round(float(g[met].mean()), 3), "min": round(float(g[met].min()), 3),
                         "max": round(float(g[met].max()), 3)} for met in ("auc", "p_at_k", "p_at_10")}
    out = {"design": {"blocks": f"k-means k={K_BLOCKS} on hex centroids (EPSG:5070), seed {SEED}",
                      "positives": "hexes with >=1 independent coffee shop", "n_hexes": len(h),
                      "n_positive_hexes": int(h.pos.sum()), "random_orderings_per_block": N_RANDOM,
                      "esri_tool_included": bool(esri_ok)},
           "per_method": summ}
    (core.RESULTS / "validation.json").write_text(json.dumps(out, indent=2))
    print(df.round(3).to_string())
    print(json.dumps(summ, indent=1))
    return out


if __name__ == "__main__":
    main(use_esri="--no-esri" not in sys.argv)
