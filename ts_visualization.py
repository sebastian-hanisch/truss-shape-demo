"""Plotly-Visualisierungen: Tragwerksplot (optimierte Form vs. Ausgangsform),
Konvergenz- und Methodenvergleich."""

import numpy as np
import plotly.graph_objects as go

from ts_model import TrussStructure

_FARBSKALA = [[0.0, "#2ca02c"], [0.6, "#f2c744"], [1.0, "#d62728"]]


def _interpoliere_farbe(u: float) -> str:
    u = max(0.0, min(1.0, u))
    for (p0, c0), (p1, c1) in zip(_FARBSKALA, _FARBSKALA[1:]):
        if p0 <= u <= p1:
            t = (u - p0) / (p1 - p0) if p1 > p0 else 0.0
            rgb0 = tuple(int(c0[i : i + 2], 16) for i in (1, 3, 5))
            rgb1 = tuple(int(c1[i : i + 2], 16) for i in (1, 3, 5))
            rgb = tuple(round(a + (b - a) * t) for a, b in zip(rgb0, rgb1))
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    return _FARBSKALA[-1][1]


def struktur_figure(
    struktur: TrussStructure, flaechen: np.ndarray, auslastung: np.ndarray, titel: str,
    referenz: TrussStructure | None = None,
) -> go.Figure:
    fig = go.Figure()

    if referenz is not None:
        for stab in referenz.staebe:
            k1, k2 = referenz.knoten[stab.knoten1], referenz.knoten[stab.knoten2]
            fig.add_trace(
                go.Scatter(
                    x=[k1.x, k2.x], y=[k1.y, k2.y], mode="lines",
                    line=dict(width=1.5, color="#bbbbbb", dash="dot"), hoverinfo="skip", showlegend=False,
                )
            )

    a_min, a_max = float(flaechen.min()), float(flaechen.max())
    spanne = max(a_max - a_min, 1e-12)
    for i, stab in enumerate(struktur.staebe):
        k1, k2 = struktur.knoten[stab.knoten1], struktur.knoten[stab.knoten2]
        breite = 2.0 + 8.0 * (flaechen[i] - a_min) / spanne
        farbe = _interpoliere_farbe(min(auslastung[i], 1.0))
        fig.add_trace(
            go.Scatter(
                x=[k1.x, k2.x], y=[k1.y, k2.y], mode="lines",
                line=dict(width=breite, color=farbe), hoverinfo="text",
                text=f"Stab {stab.id}: A={flaechen[i] * 1e4:.1f} cm², Auslastung {auslastung[i] * 100:.0f}%",
                showlegend=False,
            )
        )

    fest_x = [k.x for k in struktur.knoten if k.fest_x or k.fest_y]
    fest_y = [k.y for k in struktur.knoten if k.fest_x or k.fest_y]
    fig.add_trace(
        go.Scatter(x=fest_x, y=fest_y, mode="markers", marker=dict(symbol="triangle-up", size=13, color="#555555"), hoverinfo="skip", showlegend=False)
    )
    frei_x = [k.x for k in struktur.knoten if not (k.fest_x or k.fest_y)]
    frei_y = [k.y for k in struktur.knoten if not (k.fest_x or k.fest_y)]
    fig.add_trace(
        go.Scatter(x=frei_x, y=frei_y, mode="markers", marker=dict(size=8, color="#333333"), hoverinfo="skip", showlegend=False)
    )

    fig.update_layout(
        title=titel, xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, title="m"),
        yaxis=dict(showgrid=False, zeroline=False, title="m"), height=440,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def konvergenz_figure(verlauf: list[float]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(range(1, len(verlauf) + 1)), y=verlauf, mode="lines"))
    fig.update_layout(
        title="Konvergenz der Metaheuristik: beste Masse je Generation", xaxis_title="Generation",
        yaxis_title="Masse [kg]", height=340, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def vergleich_balken_figure(ergebnisse: dict) -> go.Figure:
    labels = list(ergebnisse.keys())
    massen = [ergebnisse[l].masse for l in labels]
    fig = go.Figure(go.Bar(x=labels, y=massen, marker_color="#1f77b4", text=[f"{m:.0f} kg" for m in massen], textposition="outside"))
    fig.update_layout(title="Gesamtmasse je Methode", yaxis_title="Masse [kg]", height=360, margin=dict(l=20, r=20, t=50, b=20))
    return fig
