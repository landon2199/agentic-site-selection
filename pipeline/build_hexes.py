"""Build the open hex table: H3 r7 hexes over the study area with ACS fields and coffee-shop counts.

    python pipeline/build_hexes.py

Inputs (data/raw/, from fetch_open_data.py) -> outputs (data/, git-ignored):
  data/hexes_open.parquet   GeoParquet, one row per hex, NO embedding values (the open fallback table)
  data/shops.parquet        GeoParquet of cleaned coffee shops with chain / independent flag and hex id

ACS tract -> hex allocation (dasymetric, 2020 block populations):
  Each 2020 census block is assigned to the hex that contains its internal point. For tract t and hex h the
  weight w(t,h) = POP100 of t's blocks in h / POP100 of t (land-area share when a tract has POP100 = 0).
  Count variables (population, people aged 25-34) are allocated with w. Median household income cannot be
  allocated as a count, so each hex gets the mean of its tracts' medians weighted by the population allocated
  from each tract (an approximation of a median, stated as such). Density = allocated population / summed
  block land area in the hex (km^2), so water and the part of an edge hex outside the county are excluded.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "study_area.yaml").read_text(encoding="utf-8"))
RAW, DATA = ROOT / "data" / "raw", ROOT / "data"
RES = CFG["h3_resolution"]
EQ_AREA = "EPSG:5070"  # CONUS Albers, for areas and distances


def county_polygon() -> gpd.GeoDataFrame:
    with zipfile.ZipFile(RAW / "cb_2023_us_county_500k.zip") as z:
        z.extractall(RAW / "cb_county")
    c = gpd.read_file(RAW / "cb_county" / "cb_2023_us_county_500k.shp")
    c = c[(c.STATEFP == CFG["state_fips"]) & (c.COUNTYFP.isin(CFG["county_fips"]))]
    return c.dissolve().to_crs(4326)


def hex_grid(county: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = county.geometry.iloc[0]
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    cells: set[str] = set()
    for p in polys:
        shape = h3.geo_to_h3shape(p.__geo_interface__)
        cells |= set(h3.h3shape_to_cells_experimental(shape, RES, contain="overlap"))
    rows = []
    for c in sorted(cells):
        boundary = [(lon, lat) for lat, lon in h3.cell_to_boundary(c)]
        lat, lon = h3.cell_to_latlng(c)
        rows.append({"h3": c, "lon": lon, "lat": lat, "geometry": Polygon(boundary)})
    g = gpd.GeoDataFrame(rows, crs=4326)
    inside = gpd.overlay(g[["h3", "geometry"]], county[["geometry"]], how="intersection").to_crs(EQ_AREA)
    frac = (inside.set_index("h3").area / g.set_index("h3").to_crs(EQ_AREA).area).rename("frac_in_area")
    g = g.merge(frac, left_on="h3", right_index=True, how="left").fillna({"frac_in_area": 0.0})
    g["area_km2_in_area"] = g.frac_in_area * g.to_crs(EQ_AREA).area.values / 1e6
    return g


def read_acs(table: str) -> pd.DataFrame:
    d = pd.read_csv(RAW / f"acs{CFG['acs_year']}_{table}.txt", sep="|", dtype=str)
    d["tract"] = d.GEO_ID.str[-11:]
    return d.set_index("tract")


def acs_tracts() -> pd.DataFrame:
    a = read_acs("b01001")
    num = lambda s: pd.to_numeric(s, errors="coerce")
    pop = num(a.B01001_E001)
    young = sum(num(a[f"B01001_E{k:03d}"]) for k in (11, 12, 35, 36))  # M 25-29, 30-34; F 25-29, 30-34
    inc = num(read_acs("b19013").B19013_E001)
    inc_moe = num(read_acs("b19013").B19013_M001)
    inc = inc.where(inc > 0)  # ACS uses negative sentinels (e.g. -666666666) for "not computed"
    return pd.DataFrame({"acs_pop": pop, "acs_pop_25_34": young, "acs_med_hh_income": inc,
                         "acs_med_hh_income_moe": inc_moe})


def allocate(hexes: gpd.GeoDataFrame) -> pd.DataFrame:
    b = pd.DataFrame(json.loads((RAW / "blocks2020.json").read_text()))
    b["tract"] = b.GEOID.str[:11]
    b["h3"] = [h3.latlng_to_cell(float(la), float(lo), RES) for la, lo in zip(b.INTPTLAT, b.INTPTLON)]
    b["POP100"] = b.POP100.fillna(0).astype(float)
    b["AREALAND"] = b.AREALAND.fillna(0).astype(float)
    tract_pop = b.groupby("tract").POP100.transform("sum")
    tract_land = b.groupby("tract").AREALAND.transform("sum")
    b["w"] = np.where(tract_pop > 0, b.POP100 / tract_pop.where(tract_pop > 0, 1),
                      b.AREALAND / tract_land.where(tract_land > 0, 1))
    w = b.groupby(["tract", "h3"]).w.sum().reset_index()
    acs = acs_tracts()
    w = w.join(acs, on="tract")
    missing = w.acs_pop.isna().sum()
    w["pop"] = w.w * w.acs_pop.fillna(0)
    w["pop_25_34"] = w.w * w.acs_pop_25_34.fillna(0)
    w["inc_x_pop"] = w.acs_med_hh_income * w["pop"]
    w["pop_inc"] = np.where(w.acs_med_hh_income.notna(), w["pop"], 0)
    g = w.groupby("h3").agg(pop=("pop", "sum"), pop_25_34=("pop_25_34", "sum"),
                            inc_x_pop=("inc_x_pop", "sum"), pop_inc=("pop_inc", "sum"),
                            n_tracts=("tract", "nunique"))
    g["med_hh_income"] = g.inc_x_pop / g.pop_inc.replace(0, np.nan)
    land = b.groupby("h3").agg(land_km2=("AREALAND", "sum"), pop2020=("POP100", "sum"), n_blocks=("GEOID", "size"))
    land["land_km2"] /= 1e6
    g = g.join(land, how="outer")
    g["share_25_34"] = g.pop_25_34 / g["pop"].replace(0, np.nan)
    g["pop_density_km2"] = g["pop"] / g.land_km2.replace(0, np.nan)
    print(f"ACS tracts: {len(acs)}; block-tract pairs without ACS row: {missing}; "
          f"ACS pop total {acs.acs_pop.sum():,.0f} vs allocated {g['pop'].sum():,.0f}")
    return g.drop(columns=["inc_x_pop", "pop_inc"])


def clean_shops() -> gpd.GeoDataFrame:
    d = pd.read_parquet(RAW / "overture_places_coffee.parquet")
    n0 = len(d)
    d = d[(d.operating_status.fillna("open") != "permanently_closed") & (d.confidence >= CFG["min_place_confidence"])]
    d = d[d.category != "internet_cafe"]
    d["name_norm"] = d.name.fillna("").str.lower().str.replace(r"[^a-z0-9&' ]", " ", regex=True).str.split().str.join(" ")
    excl = "|".join(re.escape(t) for t in CFG["exclude_name_terms"])
    d = d[~d.name_norm.str.contains(excl) & (d.name_norm != "")]
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326).to_crs(EQ_AREA)
    # de-duplicate: same normalised name within 75 m (Overture merges several sources)
    g = g.sort_values("confidence", ascending=False)
    keep, seen = [], []
    for idx, r in g.iterrows():
        dup = any(r.name_norm == n and r.geometry.distance(p) < 75 for n, p in seen)
        if not dup:
            keep.append(idx)
            seen.append((r.name_norm, r.geometry))
    g = g.loc[keep]
    key = g.name_norm.str.replace(r"\b(coffee|company|co|cafe|the)\b", "", regex=True).str.split().str.join(" ")
    multi = key.map(key.value_counts()) >= CFG["min_sites_for_chain"]
    pat = "|".join(re.escape(c.lower()) for c in CFG["chains"])
    branded = g.brand.fillna("").str.lower().str.contains(pat) | g.name_norm.str.contains(pat)
    g["is_chain"] = branded | multi
    g["h3"] = [h3.latlng_to_cell(la, lo, RES) for la, lo in zip(g.lat, g.lon)]
    print(f"places {n0} -> {len(g)} coffee shops after filters/dedupe; chains {int(g.is_chain.sum())}, "
          f"independents {int((~g.is_chain).sum())}")
    return g.to_crs(4326)[["id", "name", "brand", "category", "confidence", "address", "city", "lon", "lat",
                           "is_chain", "h3", "geometry"]]


def shop_features(hexes: gpd.GeoDataFrame, shops: gpd.GeoDataFrame) -> pd.DataFrame:
    cnt = shops.groupby("h3").agg(n_shops=("id", "size"), n_indep=("is_chain", lambda s: int((~s).sum())),
                                  n_chain=("is_chain", "sum"))
    out = hexes[["h3"]].set_index("h3").join(cnt).fillna(0).astype(int)
    ring = {h: h3.grid_disk(h, 1) for h in out.index}
    tot = out.n_shops.to_dict()
    out["n_shops_ring1"] = [sum(tot.get(n, 0) for n in ring[h]) for h in out.index]  # hex + 6 neighbours
    cen = gpd.GeoSeries(gpd.points_from_xy(hexes.lon, hexes.lat), crs=4326).to_crs(EQ_AREA)
    pts = shops.to_crs(EQ_AREA)
    xy = np.c_[pts.geometry.x, pts.geometry.y]
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(xy).query(np.c_[cen.x, cen.y])
    out["dist_nearest_shop_km"] = dist / 1000
    return out


def main() -> None:
    county = county_polygon()
    hexes = hex_grid(county)
    print(f"{len(hexes)} H3 r{RES} hexes touch {CFG['name']}")
    acs = allocate(hexes)
    shops = clean_shops()
    shops = shops[shops.h3.isin(hexes.h3)]
    feats = shop_features(hexes, shops)
    out = hexes.merge(acs, left_on="h3", right_index=True, how="left").merge(feats, left_on="h3", right_index=True)
    for c in ["pop", "pop_25_34", "land_km2", "pop2020", "n_blocks", "n_tracts"]:
        out[c] = out[c].fillna(0)
    # a block larger than a hex can put more land in a hex than the hex holds; cap at the hex's in-county area
    out["land_km2"] = np.minimum(out.land_km2, out.area_km2_in_area)
    out["pop_density_km2"] = out["pop"] / out.land_km2.where(out.land_km2 > 0.05)
    out.to_parquet(DATA / "hexes_open.parquet")
    shops.to_parquet(DATA / "shops.parquet")
    county.to_parquet(DATA / "study_area.parquet")
    print(out.describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
