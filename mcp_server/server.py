"""Site-selection MCP server: local, read-only tools over the prepared Harris County hex data.

Run over stdio:  python -m mcp_server.server   (or the `site-selection-mcp` entry point)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

# Windows: put this env's own DLL folder first so numpy/GDAL do not load mismatched system DLLs
if sys.platform == "win32":
    lib = Path(sys.prefix) / "Library" / "bin"
    if lib.exists():
        os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anyio  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

log = logging.getLogger("site_selection_mcp")
TOOL_TIMEOUT_S = 600

INSTRUCTIONS = """\
Tools for "where should an independent coffee shop open in Houston (Harris County, TX)?".
The study area is 880 H3 resolution-7 hexes (~5.2 km^2 each). Every result has `sources`, `checks` and `caveats`.
Two methods: overlay_suitability (expert-weighted overlay of ACS demographics and competition) and find_similar
(Esri USA Geodemographic Embeddings: hexes most like those that already hold independent shops). Use validate to
say how well each method recovers held-out shops before you express confidence, and report method agreement.
Cite the engine and sources, mention any warn/fail check, and never invent numbers a tool did not return.
There is no sales, rent, lease, foot-traffic or survival data: say so if a question needs it."""

mcp = MCPServer(
    name="site-selection",
    title="Agentic Site Selection (Houston coffee)",
    description="Overlay suitability, Esri geodemographic-embedding similarity, hot spots and validation for Harris County hexes.",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)

LOCAL = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WEB = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)


async def _run(fn: Callable[..., dict], *args: Any, **kw: Any) -> dict:
    from . import tools

    try:
        with anyio.fail_after(TOOL_TIMEOUT_S):
            return await anyio.to_thread.run_sync(lambda: fn(*args, **kw), abandon_on_cancel=True)
    except TimeoutError:
        raise ToolError(f"Timed out after {TOOL_TIMEOUT_S} s.") from None
    except tools.ToolInputError as exc:
        raise ToolError(str(exc)) from None
    except Exception as exc:  # pragma: no cover
        log.exception("unexpected error")
        raise ToolError(f"Unexpected error ({type(exc).__name__}): {exc}") from None


@mcp.tool(title="Geocode an address", annotations=WEB)
async def geocode(address: str) -> dict:
    """Geocode a one-line U.S. street address with the free U.S. Census Geocoder (no ArcGIS credits).
    Returns lon/lat, the H3 r7 hex id and whether it is inside the Harris County study area.
    Street addresses only; business names and landmarks do not match."""
    from .tools import geocode as impl

    return await _run(impl, address)


@mcp.tool(title="Profile one hex", annotations=LOCAL)
async def hex_profile(h3_id: str | None = None, lon: float | None = None, lat: float | None = None) -> dict:
    """Everything known about one hex: population, share aged 25-34, median household income (approx.),
    density, coffee shops (chains / independents, names if available), overlay score and sub-scores,
    embedding similarity, percentiles, top-10% agreement class and hot-spot z. Give h3_id or lon+lat (WGS84)."""
    from .tools import hex_profile as impl

    return await _run(impl, h3_id=h3_id, lon=lon, lat=lat)


@mcp.tool(title="Weighted-overlay suitability", annotations=LOCAL)
async def overlay_suitability(weights: dict[str, float] | None = None, top_n: int = 10,
                              exclude_existing_independents: bool = False) -> dict:
    """Method 1. Rescale four criteria to 1-10 and take a weighted sum: pop_density_km2 (log, increasing),
    share_25_34 (increasing), med_hh_income (increasing), n_shops_ring1 (coffee shops in hex + 6 neighbours,
    decreasing). Default weights 0.30 / 0.25 / 0.20 / 0.25; pass `weights` to override any of them (renormalised).
    Hexes with < 500 residents or < 1 km^2 land are masked. Returns the top_n (1-50) hexes with sub-scores."""
    from .tools import overlay_suitability as impl

    return await _run(impl, weights=weights, top_n=top_n, exclude_existing_independents=exclude_existing_independents)


@mcp.tool(title="Find similar hexes (embeddings)", annotations=LOCAL)
async def find_similar(query: list[str] | str = "independent_shops", top_n: int = 10, exclude_existing: bool = True,
                       engine: str = "auto") -> dict:
    """Method 2. Rank hexes by cosine similarity of Esri's USA Geodemographic Embeddings to a query.
    query = "independent_shops" (all hexes that hold an independent coffee shop) or a list of H3 r7 ids.
    exclude_existing drops query hexes and hexes that already have an independent shop ("untapped look-alikes").
    engine: auto (precomputed Esri run for the default query, numpy for custom queries), arcpy (live
    FindSimilarFeaturesUsingEmbeddings, needs ArcGIS Pro Advanced; ~1 min), numpy, or open_proxy (no Esri data)."""
    from .tools import find_similar as impl

    return await _run(impl, query=query, top_n=top_n, exclude_existing=exclude_existing, engine=engine)


@mcp.tool(title="Hot spots (Gi*)", annotations=LOCAL)
async def hot_spots(field: str = "overlay", top_n: int = 10) -> dict:
    """Getis-Ord Gi* hot-spot analysis of one hex field (hex + 6 neighbours). Fields: overlay, emb, pop,
    pop_density_km2, share_25_34, med_hh_income, n_shops, n_indep, n_chain, dist_nearest_shop_km.
    Returns counts of hot/cold hexes at 95% and the hottest top_n hexes."""
    from .tools import hot_spots as impl

    return await _run(impl, field=field, top_n=top_n)


@mcp.tool(title="Validation results", annotations=LOCAL)
async def validate(method: str = "all") -> dict:
    """Spatially blocked hold-out (5 k-means blocks): hide a block's shops, rebuild from the rest, and score how
    well each method ranks the held-out independent-shop hexes (AUC, precision@k) against a random baseline.
    method: all, overlay, overlay_demand_only, embeddings (= embeddings_esri), embeddings_numpy, random."""
    from .tools import validate as impl

    return await _run(impl, method=method)


@mcp.resource("catalog://datasets", name="datasets", title="Datasets, licenses and capabilities",
              description="What data this server uses, its license, and which tiers are available here.",
              mime_type="application/json")
def datasets() -> str:
    from . import tools

    return json.dumps({
        "capabilities_here": tools.capabilities(),
        "datasets": {
            "acs": {"source": tools.SRC_ACS, "license": "U.S. Census Bureau, public domain"},
            "shops": {"source": tools.SRC_SHOPS, "license": "Overture Maps Places: CDLA Permissive 2.0 (see Overture attribution)"},
            "embeddings": {"source": tools.SRC_EMB, "license": "Esri Master License Agreement; subscriber content; "
                           "only derived scores are distributed, never the vectors"},
            "geocoder": {"source": "U.S. Census Geocoder", "license": "public domain"},
        }}, indent=2)


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
