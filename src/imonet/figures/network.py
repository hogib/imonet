"""Network figures.

- network_map: the California stations, normal and held-out.
- thinning: the normal network at each evaluation density, and the
  nearest-neighbour distance distribution of each.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imonet.cube import HELD_OUT, NORMAL, Cube, pairwise_km
from imonet.evaluate import DENSITIES, density_name, network
from imonet.figures.style import BLUE, BLUE_LIGHT, DENSITY_SHADES, HALF, ORANGE, STYLE, WIDTH, save
from imonet.network import nearest_km


def _geographic(ax, lat0: float) -> None:
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")


def network_map(cube: Cube, out: Path) -> Path:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(HALF + 0.4, HALF + 0.6))
        n, h = cube.stations(NORMAL), cube.stations(HELD_OUT)
        ax.scatter(cube.lon[n], cube.lat[n], s=4, color=BLUE, lw=0, label=f"normal ({len(n)})")
        ax.scatter(cube.lon[h], cube.lat[h], s=12, color=ORANGE, lw=0.6, edgecolor="white",
                   label=f"held-out ({len(h)})")
        _geographic(ax, float(np.median(cube.lat)))
        ax.legend(loc="upper right", markerscale=1.6, handletextpad=0.2)
        return save(fig, out, "network_map")


def thinning(cube: Cube, out: Path, seed: int = 0) -> Path:
    """One map per evaluation density (kept stations over the full network),
    and the nearest-neighbour distances of each."""
    nets = {density_name(d): network(cube, d, seed) for d in DENSITIES}
    normal = cube.stations(NORMAL)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH, 4.6))
        grid = fig.add_gridspec(2, len(nets), height_ratios=[1.6, 1.0], hspace=0.35)
        lat0 = float(np.median(cube.lat))
        for i, (name, net) in enumerate(nets.items()):
            ax = fig.add_subplot(grid[0, i])
            ax.scatter(cube.lon[normal], cube.lat[normal], s=1.5, color=BLUE_LIGHT, lw=0)
            ax.scatter(cube.lon[net], cube.lat[net], s=4, color=BLUE, lw=0)
            ax.set_aspect(1 / np.cos(np.radians(lat0)))
            ax.set_title(f"{name}: {len(net)} stations", fontsize=8)
            ax.tick_params(labelsize=6.5)
            if i:
                ax.set_yticklabels([])
        ax = fig.add_subplot(grid[1, :])
        bins = np.geomspace(1, 300, 40)
        for i, (name, net) in enumerate(nets.items()):
            nn = nearest_km(pairwise_km(cube.lat[net], cube.lon[net]))
            hist, edges = np.histogram(nn, bins=bins, density=True)
            centre = np.sqrt(edges[1:] * edges[:-1])
            shade = DENSITY_SHADES[i]
            ax.plot(centre, hist / hist.max(), color=shade, label=f"{name} (median {np.median(nn):.0f} km)")
        ax.set_xscale("log")
        ax.set_xlabel("Distance to nearest station (km)")
        ax.set_ylabel("Relative frequency")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2)
        return save(fig, out, "thinning")
