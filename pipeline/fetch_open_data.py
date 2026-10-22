"""Download the open inputs for the study area into data/raw/ (git-ignored).

    python pipeline/fetch_open_data.py            # everything
    python pipeline/fetch_open_data.py places     # only Overture places

Sources (all free, no key):
  * ACS 2020-2024 5-year, tables B01001 / B01003 / B19013, from the Census Bureau's table-based
    summary files (the api.census.gov endpoint now answers "Missing Key", so the files are used instead).
  * 2020 Census block populations (POP100) and internal points from TIGERweb (Tracts_Blocks layer 2).
  * County boundary: Census cartographic boundary file cb_2023_us_county_500k.
  * Coffee shops: Overture Maps Places, read with DuckDB straight from the public S3 release.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "study_area.yaml").read_text())
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

STATE, COUNTIES = CFG["state_fips"], CFG["county_fips"]
UA = {"User-Agent": "agentic-site-selection/0.1 (portfolio project)"}


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def acs() -> None:
    year = CFG["acs_year"]
    base = f"https://www2.census.gov/programs-surveys/acs/summary_file/{year}/table-based-SF/data/5YRData"
    prefixes = tuple(f"1400000US{STATE}{c}" for c in COUNTIES)
    for table in ("b01001", "b01003", "b19013"):
        out = RAW / f"acs{year}_{table}.txt"
        if out.exists():
            print("have", out.name)
            continue
        req = urllib.request.Request(f"{base}/acsdt5y{year}-{table}.dat", headers=UA)
        keep = []
        with urllib.request.urlopen(req, timeout=600) as r:
            for i, line in enumerate(io.TextIOWrapper(r, encoding="utf-8")):
                if i == 0 or line.startswith(prefixes):
                    keep.append(line)
        out.write_text("".join(keep), encoding="utf-8")
        print(f"{out.name}: {len(keep) - 1} tracts")


def blocks() -> None:
    out = RAW / "blocks2020.json"
    if out.exists():
        print("have", out.name)
        return
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/2/query"
    rows: list[dict] = []
    for c in COUNTIES:
        offset = 0
        while True:
            q = urllib.parse.urlencode({
                "where": f"STATE='{STATE}' AND COUNTY='{c}'",
                "outFields": "GEOID,POP100,HU100,AREALAND,AREAWATER,INTPTLAT,INTPTLON",
                "returnGeometry": "false", "f": "json", "orderByFields": "OBJECTID",
                "resultOffset": offset, "resultRecordCount": 5000,
            })
            d = json.loads(_get(f"{url}?{q}"))
            feats = [f["attributes"] for f in d.get("features", [])]
            rows += feats
            offset += len(feats)
            if not feats or not d.get("exceededTransferLimit"):
                break
        print(f"county {c}: {offset} blocks")
    out.write_text(json.dumps(rows))


def county() -> None:
    out = RAW / "cb_2023_us_county_500k.zip"
    if not out.exists():
        out.write_bytes(_get("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"))
    print("have", out.name)


def places() -> None:
    import duckdb

    out = RAW / "overture_places_coffee.parquet"
    if out.exists():
        print("have", out.name)
        return
    w, s, e, n = CFG["bbox"]
    rel = CFG["overture_release"]
    con = duckdb.connect()
    for ext in ("httpfs", "spatial"):
        con.install_extension(ext)
        con.load_extension(ext)
    con.execute("SET s3_region='us-west-2'")
    cats = ", ".join(f"'{c}'" for c in CFG["coffee_categories"])
    sql = f"""
    COPY (
      SELECT id, names.primary AS name, brand.names.primary AS brand,
             taxonomy.primary AS category, basic_category, array_to_string(taxonomy.hierarchy, '>') AS hierarchy,
             confidence, operating_status,
             addresses[1].freeform AS address, addresses[1].locality AS city,
             ST_X(geometry) AS lon, ST_Y(geometry) AS lat
      FROM read_parquet('s3://overturemaps-us-west-2/release/{rel}/theme=places/type=place/*', hive_partitioning=1)
      WHERE bbox.xmin BETWEEN {w} AND {e} AND bbox.ymin BETWEEN {s} AND {n}
        AND (taxonomy.primary IN ({cats}) OR basic_category IN ({cats})
             OR list_has_any(taxonomy.hierarchy, [{cats}]))
    ) TO '{out.as_posix()}' (FORMAT parquet)
    """
    con.execute(sql)
    print(out.name, con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0], "places")


if __name__ == "__main__":
    import urllib.parse  # noqa: F401  (used in blocks)

    steps = {"acs": acs, "blocks": blocks, "county": county, "places": places}
    for name in sys.argv[1:] or steps:
        steps[name]()
