"""Build docs/hero.png: overlay score, embedding similarity and their top-10% agreement (from docs/data/hexes.geojson)."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
g = gpd.read_file(HERE / "data" / "hexes.geojson").to_crs("EPSG:5070")
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
g.plot("overlay", ax=ax[0], cmap="Oranges", scheme="quantiles", k=5, linewidth=0.1, edgecolor="white")
ax[0].set_title("Method 1: weighted overlay (quintiles)")
g.plot("emb", ax=ax[1], cmap="Blues", scheme="quantiles", k=5, linewidth=0.1, edgecolor="white")
ax[1].set_title("Method 2: Esri embedding similarity (quintiles)")
colors = {"both": "#b4432f", "overlay only": "#e9a23b", "embeddings only": "#3b7dd8", "neither": "#e4e4df"}
g.plot(color=g.agreement.map(colors), ax=ax[2], linewidth=0.1, edgecolor="white")
ax[2].set_title("Top 10% agreement (Jaccard 0.023)")
ax[2].legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=9, frameon=False)
for a in ax:
    a.set_axis_off()
fig.suptitle("Where should an independent coffee shop open? 880 H3 r7 hexes, Harris County, TX", fontsize=13)
fig.text(0.5, 0.02, "Data: ACS 2020-2024, 2020 Census blocks, Overture Places 2026-10-21, Esri USA Geodemographic Embeddings 2026 (beta)",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(HERE / "hero.png", dpi=110)
print("wrote docs/hero.png")
