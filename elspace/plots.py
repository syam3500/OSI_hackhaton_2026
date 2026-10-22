from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import numpy as np

from elspace.chemistry import DESCRIPTORS

BLUE = "#002FA7"
PALETTE = [BLUE, "#7F91C0", "#B9BDC7", "#DCDDE2", "#61718F", "#303F5C"]
STATUS_COLORS = {"Measured": BLUE, "Represented": "#7F91C0", "Unsupported": "#C7CAD2", "Unresolved": "#E5E6EA"}


def chemical_map(frame: pd.DataFrame, clap: pd.DataFrame, target: dict, neighbours: pd.DataFrame,
                 color_by: str, dimensions: str, projection: str, show_clap: bool,
                 descriptor_axes: list[str] | None = None) -> go.Figure:
    axes = ["map_x", "map_y", "map_z"] if projection == "Fingerprint PCA" else list(descriptor_axes or ["mol_weight", "logp", "tpsa"])
    count = 3 if dimensions == "3D" else 2
    axes = axes[:count]
    if len(axes) != count or len(set(axes)) != count or (projection != "Fingerprint PCA" and any(axis not in DESCRIPTORS for axis in axes)):
        raise ValueError("Choose a different valid descriptor for each plotted axis.")
    frame = frame.dropna(subset=axes)
    clap = clap.dropna(subset=axes)
    if len(neighbours):
        neighbours = neighbours.dropna(subset=axes)
    figure = go.Figure()
    trace_type = go.Scatter3d if dimensions == "3D" else go.Scattergl

    def trace(group, name, marker):
        arguments = {
            "x": group[axes[0]], "y": group[axes[1]], "mode": "markers", "name": name,
            "customdata": [[row.compound_id, escape(str(row.name)), escape(str(row.cas_raw))] for row in group.itertuples()],
            "marker": marker,
            "hovertemplate": "<b>%{customdata[1]}</b><br>CAS: %{customdata[2]}<br>%{customdata[0]}<extra>%{fullData.name}</extra>",
        }
        if dimensions == "3D":
            arguments["z"] = group[axes[2]]
        return trace_type(**arguments)

    for index, (name, group) in enumerate(frame.groupby(color_by, sort=True)):
        figure.add_trace(trace(group, str(name), {"size": 5 if dimensions == "2D" else 3,
                                                 "color": STATUS_COLORS.get(name, PALETTE[index % len(PALETTE)]), "opacity": 0.65}))
    if show_clap and len(clap):
        figure.add_trace(trace(clap, "FDA CLAP panel", {"size": 9 if dimensions == "2D" else 5,
                                                        "symbol": "diamond", "color": BLUE, "opacity": 0.95}))
    if target.get("smiles_std") and all(pd.notna(target.get(axis)) for axis in axes):
        if len(neighbours):
            links = {axis: [] for axis in axes}
            for _, row in neighbours.iterrows():
                for axis in axes:
                    links[axis].extend([target[axis], row[axis], None])
            line = {"x": links[axes[0]], "y": links[axes[1]], "mode": "lines", "name": "Closest CLAP structures",
                    "line": {"color": "#8992A7", "width": 1}, "hoverinfo": "skip", "showlegend": False}
            if dimensions == "3D":
                line["z"] = links[axes[2]]
            figure.add_trace(trace_type(**line))
            figure.add_trace(trace(neighbours, "Nearest CLAP neighbours", {"size": 12 if dimensions == "2D" else 6,
                                                                          "color": "#202128", "symbol": "diamond"}))
        figure.add_trace(trace(pd.DataFrame([target]), "Selected compound", {"size": 17 if dimensions == "2D" else 8,
                                                                            "color": "#202128", "symbol": "cross"}))
    labels = ["Fingerprint PC1", "Fingerprint PC2", "Fingerprint PC3"] if projection == "Fingerprint PCA" else [DESCRIPTORS[axis] for axis in axes]
    x_label, y_label = labels[:2]
    z_label = labels[2] if dimensions == "3D" else ""
    axis_style = {"gridcolor": "#ECEEF2", "zerolinecolor": "#ECEEF2", "showline": False}
    figure.update_layout(
        height=550, margin={"l": 8, "r": 8, "t": 12, "b": 0}, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"family": "Helvetica Neue, Arial, sans-serif", "color": "#454A56", "size": 12},
        legend={"orientation": "h", "y": -0.13, "x": 0}, hoverlabel={"bgcolor": "white"},
        xaxis={"title": x_label, **axis_style}, yaxis={"title": y_label, **axis_style},
        uirevision=f"{dimensions}-{projection}-{'-'.join(axes)}", clickmode="event+select",
        scene={"xaxis": {"title": x_label, **axis_style}, "yaxis": {"title": y_label, **axis_style},
               "zaxis": {"title": z_label, **axis_style}, "bgcolor": "#FFFFFF"},
    )
    return figure


def spectrum_plot(spectrum: dict, title: str) -> go.Figure:
    from elspace.spectra import parse_peaks

    masses, intensities = parse_peaks(spectrum["mzs"], spectrum["intensities"])
    relative = intensities / intensities.max() * 100
    stick_x, stick_y = [], []
    for mass, intensity in zip(masses, relative):
        stick_x.extend([float(mass), float(mass), None])
        stick_y.extend([0, float(intensity), None])
    figure = go.Figure(go.Scatter(x=stick_x, y=stick_y, mode="lines", line={"color": BLUE, "width": 1.5},
                                 hoverinfo="skip", showlegend=False))
    labeled_peaks = set(np.argsort(relative)[-5:])
    labels = [f"{mass:.4f}".rstrip("0").rstrip(".") if index in labeled_peaks else "" for index, mass in enumerate(masses)]
    figure.add_trace(go.Scatter(x=masses, y=relative, mode="markers+text", text=labels, textposition="top center",
                               marker={"size": 4, "color": BLUE}, showlegend=False,
                               hovertemplate="m/z %{x:.5f}<br>Relative intensity %{y:.1f}%<extra></extra>"))
    precursor = spectrum.get("precursor_mz")
    if precursor is not None and pd.notna(precursor):
        figure.add_vline(x=float(precursor), line_dash="dash", line_color="#8992A7", annotation_text="Precursor")
    figure.update_layout(title={"text": escape(title), "font": {"size": 15}}, height=320,
                         margin={"l": 45, "r": 20, "t": 55, "b": 40}, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                         font={"family": "Helvetica Neue, Arial, sans-serif", "color": "#454A56"},
                         xaxis={"title": "m/z", "gridcolor": "#ECEEF2"},
                         yaxis={"title": "Relative intensity (%)", "range": [0, 118], "gridcolor": "#ECEEF2"})
    return figure
