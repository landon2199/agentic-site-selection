"""Offline tests: core maths and the MCP surface over stdio (needs results/ only, not data/)."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sitesel import core  # noqa: E402


def test_auc_and_precision():
    assert core.auc(np.array([3, 2, 1, 0]), np.array([1, 1, 0, 0])) == 1.0
    assert core.auc(np.array([0, 1, 2, 3]), np.array([1, 1, 0, 0])) == 0.0
    assert core.precision_at_k(np.array([3, 2, 1, 0]), np.array([1, 0, 1, 0]), 2) == 0.5


def test_rescale_bounds_and_direction():
    x = pd.Series(np.arange(100.0))
    up = core.rescale(x, "linear", "increasing", 5, 95)
    down = core.rescale(x, "linear", "decreasing", 5, 95)
    assert up.min() == 1 and up.max() == 10 and down.iloc[0] == 10


def test_similar_numpy_mean_query():
    emb = pd.DataFrame(np.eye(3), index=["a", "b", "c"])
    s = core.similar_numpy(emb, ["a", "b"])
    assert s["a"] == pytest.approx(s["b"]) and s["c"] == pytest.approx(0)


def test_mcp_stdio_tools():
    from mcp import Client, StdioServerParameters

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT))
        async with Client(params, read_timeout_seconds=120) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == {"geocode", "hex_profile", "overlay_suitability", "find_similar", "hot_spots", "validate"}
            r = await c.call_tool("overlay_suitability", {"top_n": 3})
            data = r.structured_content or json.loads(r.content[0].text)
            data = data.get("result", data)
            assert len(data["top"]) == 3 and all(ch["status"] != "fail" for ch in data["checks"])
            bad = await c.call_tool("hex_profile", {"lon": 29.76, "lat": -95.37})
            assert bad.is_error and "swapped" in bad.content[0].text

    asyncio.run(run())
