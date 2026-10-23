from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go

BLUE = "#002FA7"
PALETTE = [BLUE, "#7F91C0", "#B9BDC7", "#DCDDE2", "#61718F", "#303F5C"]
STATUS_COLORS = {"Measured": BLUE, "Represented": "#7F91C0", "Unsupported": "#C7CAD2", "Unresolved": "#E5E6EA"}


def chemical_map(frame: pd.DataFrame, clap: pd.DataFrame, target: dict, neighbours: pd.DataFrame,
                 color_by: str, dimensions: str, projection: str, show_clap: bool) -> go.Figure:
    axes = ["map_x", "map_y", "map_z"] if projection == "Fingerprint PCA" else ["mol_weight", "logp", "tpsa"]
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
    if target.get("smiles_std") and pd.notna(target.get(axes[0])):
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
    x_label, y_label, z_label = ("Fingerprint PC1", "Fingerprint PC2", "Fingerprint PC3") if projection == "Fingerprint PCA" else ("Molecular weight (g/mol)", "Crippen logP", "TPSA (Å²)")
    axis_style = {"gridcolor": "#ECEEF2", "zerolinecolor": "#ECEEF2", "showline": False}
    figure.update_layout(
        height=550, margin={"l": 8, "r": 8, "t": 12, "b": 0}, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"family": "Helvetica Neue, Arial, sans-serif", "color": "#454A56", "size": 12},
        legend={"orientation": "h", "y": -0.13, "x": 0}, hoverlabel={"bgcolor": "white"},
        xaxis={"title": x_label, **axis_style}, yaxis={"title": y_label, **axis_style},
        uirevision=f"{dimensions}-{projection}", clickmode="event+select",
        scene={"xaxis": {"title": x_label, **axis_style}, "yaxis": {"title": y_label, **axis_style},
               "zaxis": {"title": z_label, **axis_style}, "bgcolor": "#FFFFFF"},
    )
    return figure
