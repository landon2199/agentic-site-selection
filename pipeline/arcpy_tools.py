"""ArcPy steps (need ArcGIS Pro 3.7 Advanced + an ArcGIS organizational sign-in). Run with Pro's Python:

    "C:/Program Files/ArcGIS/Pro/bin/Python/envs/arcgispro-py3/python.exe" pipeline/arcpy_tools.py export
    "...python.exe" pipeline/arcpy_tools.py hotspots <values.csv> <out_name>
    "...python.exe" pipeline/arcpy_tools.py similar <query_ids.txt> <out_name> [<query2.txt> <out2> ...]

export   Copies ONLY the study-area hexes of Esri's "USA Geodemographic Embeddings 2026" (subscriber content,
         Living Atlas item 6637a530b840446285a4383627a13f1f) into a LOCAL file geodatabase,
         data/esri/embeddings.gdb/hex_emb, using the portal sign-in that ArcGIS Pro is using. Nothing is
         hosted, no credits are spent. It also writes data/esri/embeddings_local.parquet (h3 id + 256 floats)
         so the numpy fallback can use the same vectors. Both stay in data/ (git-ignored, Esri MLA).
hotspots Runs arcpy.stats.HotSpots (Getis-Ord Gi*) on <values.csv> (h3, value) joined to the local hexes,
         contiguity edges+corners; writes data/esri/<out_name>.csv (h3, GiZScore, GiPValue). Spatial Statistics
         is in every license level; this is the cross-check for the numpy Gi* in sitesel/core.py.
similar  Runs arcpy.geoai.FindSimilarFeaturesUsingEmbeddings with the hexes listed in <query_ids.txt> as the
         query features and threshold 0 (score every hex); writes data/esri/<out_name>.csv (h3, score).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import arcpy

ROOT = Path(__file__).resolve().parents[1]
ESRI = ROOT / "data" / "esri"
GDB = ESRI / "embeddings.gdb"
SERVICE = ("https://services8.arcgis.com/peDZJliSvYims39Q/arcgis/rest/services/"
           "USA_Geodemographic_Embeddings_2026_V2_Beta_View/FeatureServer/0")
FC = str(GDB / "hex_emb")
arcpy.env.overwriteOutput = True


def _ids() -> list[str]:
    import pandas as pd

    return pd.read_parquet(ROOT / "data" / "hexes_open.parquet", columns=["h3"]).h3.tolist()


def export() -> None:
    ESRI.mkdir(parents=True, exist_ok=True)
    if not arcpy.Exists(str(GDB)):
        arcpy.management.CreateFileGDB(str(ESRI), "embeddings.gdb")
    ids = _ids()
    t = time.time()
    parts = []
    for i in range(0, len(ids), 200):  # keep each WHERE clause short
        chunk = ",".join(f"'{h}'" for h in ids[i:i + 200])
        out = f"memory/part{i // 200}"
        arcpy.conversion.ExportFeatures(SERVICE, out, where_clause=f"ID IN ({chunk})")
        parts.append(out)
        print(f"  chunk {i // 200}: {arcpy.management.GetCount(out)[0]} rows", flush=True)
    arcpy.management.Merge(parts, FC)
    n = int(arcpy.management.GetCount(FC)[0])
    print(f"exported {n} of {len(ids)} study-area hexes to {FC} in {time.time() - t:.0f} s", flush=True)

    import numpy as np
    import pandas as pd

    emb_cols = [f"emb_{k}" for k in range(256)]
    rows = [r for r in arcpy.da.SearchCursor(FC, ["ID"] + emb_cols)]
    df = pd.DataFrame(rows, columns=["h3"] + emb_cols)
    df[emb_cols] = df[emb_cols].astype(np.float32)
    df.to_parquet(ESRI / "embeddings_local.parquet")
    # Esri's own ACS fields on the hexes, used only as a cross-check of our tract->hex allocation
    acs = ["ID", "ACSTOTPOP", "ACSTOTHH", "ACSMEDHINC", "AREA", "ADMIN3_NAME"]
    pd.DataFrame([r for r in arcpy.da.SearchCursor(FC, acs)], columns=acs).to_csv(ESRI / "esri_hex_acs.csv", index=False)
    print("wrote embeddings_local.parquet and esri_hex_acs.csv", flush=True)


def similar(query_file: str, out_name: str) -> None:
    q = [s.strip() for s in Path(query_file).read_text().split() if s.strip()]
    where = "ID IN (" + ",".join(f"'{h}'" for h in q) + ")"
    qfc = "memory/query"
    arcpy.conversion.ExportFeatures(FC, qfc, where_clause=where)
    out_fc = str(GDB / f"sim_{out_name}")
    t = time.time()
    res = arcpy.geoai.FindSimilarFeaturesUsingEmbeddings(FC, qfc, out_fc, 0)
    secs = time.time() - t
    for m in range(res.messageCount):
        print("  msg:", res.getMessage(m), flush=True)
    fields = [f.name for f in arcpy.ListFields(out_fc)]
    print("output fields:", [f for f in fields if not f.startswith("emb_")], flush=True)
    score_field = next((f for f in fields if "SIM" in f.upper() or "SCORE" in f.upper()), None)
    if score_field is None:
        raise SystemExit(f"no similarity field found in {fields}")
    import pandas as pd

    rows = [r for r in arcpy.da.SearchCursor(out_fc, ["ID", score_field])]
    pd.DataFrame(rows, columns=["h3", "esri_score"]).to_csv(ESRI / f"{out_name}.csv", index=False)
    print(f"{len(q)} query hexes -> {len(rows)} scored hexes via field {score_field} in {secs:.1f} s", flush=True)


def hotspots(values_csv: str, out_name: str) -> None:
    import pandas as pd

    vals = pd.read_csv(values_csv)
    fc = "memory/hs_in"
    arcpy.management.CopyFeatures(FC, fc)
    arcpy.management.DeleteField(fc, ["ID"], "KEEP_FIELDS")
    arcpy.management.AddField(fc, "VAL", "DOUBLE")
    lut = dict(zip(vals.h3, vals.value))
    with arcpy.da.UpdateCursor(fc, ["ID", "VAL"]) as cur:
        for row in cur:
            row[1] = float(lut.get(row[0], 0.0))
            cur.updateRow(row)
    out = "memory/hs_out"
    arcpy.stats.HotSpots(fc, "VAL", out, "CONTIGUITY_EDGES_CORNERS", "EUCLIDEAN_DISTANCE", "NONE")
    oid = {r[0]: r[1] for r in arcpy.da.SearchCursor(fc, ["OID@", "ID"])}
    rows = [(oid[r[0]], r[1], r[2]) for r in arcpy.da.SearchCursor(out, ["SOURCE_ID", "GiZScore", "GiPValue"])]
    pd.DataFrame(rows, columns=["h3", "gi_z", "gi_p"]).to_csv(ESRI / f"{out_name}.csv", index=False)
    print(f"hot spots for {len(rows)} hexes -> data/esri/{out_name}.csv", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        export()
    elif cmd == "hotspots":
        hotspots(sys.argv[2], sys.argv[3])
    elif cmd == "similar":  # similar q1.txt out1 [q2.txt out2 ...] in one ArcPy session
        args = sys.argv[2:]
        for q, o in zip(args[::2], args[1::2]):
            similar(q, o)
    else:
        raise SystemExit(__doc__)
