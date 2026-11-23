#!/usr/bin/env python3
"""
Plot ECEF ground-truth trajectory on an interactive OpenStreetMap background.

Expected input: repeated records with numbered fields, e.g.
1 - "point3" [string]
2 - 123.456 [s]
3 -  ... ECEF-X ...
4 -  ... ECEF-Y ...
5 -  ... ECEF-Z ...
6 - 0
...
14 - 0

Usage:
    python plot_ecef_map.py ground_truth.txt
    python plot_ecef_map.py ground_truth.txt --output trajectory.html
    python plot_ecef_map.py ground_truth.txt --marker-every 100

Dependencies:
    python -m pip install numpy pyproj folium
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import folium
import numpy as np
from pyproj import Transformer


# ECEF (WGS84) -> geodetic longitude, latitude, altitude (WGS84)
ECEF_TO_GEODETIC = Transformer.from_crs(
    "EPSG:4978", "EPSG:4979", always_xy=True
)

FIELD_RE = re.compile(
    r'^\s*(?:(\d+)\s*[-:]\s*)?(.*?)\s*$'
)
NUMBER_RE = re.compile(
    r"[-+]?(?:\\d+\\.\\d*|\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?"
)


def parse_number(text: str) -> float:
    """Extract the first numeric value from a field's value text."""
    match = NUMBER_RE.search(text)
    if not match:
        raise ValueError(f"No numeric value found in field: {text!r}")
    return float(match.group(0))


def read_records(filename: Path):
    timestamps = []
    xyz = []

    with filename.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] == 'point3':
                try:
                    time_value = float(parts[1])
                    x_loc = float(parts[2])
                    y_loc = float(parts[3])
                    z_loc = float(parts[4])
                    xyz.append(np.array([x_loc, y_loc, z_loc]))
                    timestamps.append(time_value)
                except:
                    print("Skipping line due to invalid number format: %s", line.strip())
                    continue

    timestamps = np.asarray(timestamps, dtype=float)
    xyz = np.asarray(xyz, dtype=float)

    if len(timestamps) != len(xyz):
        raise ValueError(
            f"Timestamp count ({len(timestamps)}) does not match "
            f"position count ({len(xyz)})."
        )

    return timestamps, xyz


def make_map(timestamps, xyz, output: Path, marker_every: int = 0):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon, lat, alt = ECEF_TO_GEODETIC.transform(x, y, z)
    coords = np.column_stack((lat, lon))
    center = np.mean(coords, axis=0)

    m = folium.Map(
        location=center.tolist(),
        zoom_start=17,
        tiles=None,  # add explicit tile layer below
        control_scale=True,
    )

    # OSM Standard tiles, correct HTTPS endpoint, visible attribution.
    # Browser fetches tiles for the active viewport; no prefetch/download.
    folium.TileLayer(
        tiles="CartoDB positron",
        # name="OpenStreetMap",
        # attr='© <a href="https://www.openstreetmap.org/copyright">'
        #      'OpenStreetMap contributors</a>',
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(m)

    folium.PolyLine(
        coords.tolist(),
        color="red",
        weight=4,
        opacity=0.85,
        tooltip="Ground-truth trajectory",
    ).add_to(m)

    folium.Marker(
        coords[0].tolist(),
        tooltip="Start",
        popup=f"Start<br>Time: {timestamps[0]:.3f} s<br>Altitude: {alt[0]:.2f} m",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    folium.Marker(
        coords[-1].tolist(),
        tooltip="End",
        popup=f"End<br>Time: {timestamps[-1]:.3f} s<br>Altitude: {alt[-1]:.2f} m",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    if marker_every > 0:
        for i in range(0, len(coords), marker_every):
            folium.CircleMarker(
                location=coords[i].tolist(),
                radius=3,
                color="blue",
                fill=True,
                fill_opacity=0.7,
                tooltip=f"Point {i}",
                popup=(
                    f"Index: {i}<br>Time: {timestamps[i]:.3f} s<br>"
                    f"Lat: {lat[i]:.8f}<br>Lon: {lon[i]:.8f}<br>"
                    f"Alt: {alt[i]:.2f} m<br>"
                    f"ECEF X: {x[i]:.3f} m<br>"
                    f"ECEF Y: {y[i]:.3f} m<br>"
                    f"ECEF Z: {z[i]:.3f} m"
                ),
            ).add_to(m)

    m.fit_bounds(
        [[float(np.min(lat)), float(np.min(lon))],
         [float(np.max(lat)), float(np.max(lon))]]
    )
    folium.LayerControl(collapsed=False).add_to(m)

    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    print(f"Parsed {len(xyz)} trajectory points.")
    print(f"Map saved to: {output.resolve()}")
    print("Open the HTML file in a browser to interact with the map.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a numbered ECEF ground-truth file on an OSM map."
    )
    parser.add_argument("input", type=Path, help="Input text file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("trajectory_map.html"),
        help="Output HTML map (default: trajectory_map.html)"
    )
    parser.add_argument(
        "--marker-every", type=int, default=0,
        help="Add a popup marker every N points (0 disables sampled markers)."
    )
    args = parser.parse_args()

    timestamps, xyz = read_records(args.input)
    make_map(timestamps, xyz, args.output, args.marker_every)


if __name__ == "__main__":
    main()
