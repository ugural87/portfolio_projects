from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import plotly.graph_objects as go


if __name__ == "__main__":
    nodes = {
        "Yield panel": (0, 4, "7 maturities × 4 causal channels"),
        "2D CNN": (1, 4, "3×3 and 7×3 branches; local time/maturity patterns"),
        "Price Transformer": (2, 4, "Daily tokens; frozen pre-1993 backbone"),
        "Price tokens": (3, 4, "Up to 60 valid intermeeting days"),
        "Rate token": (1, 2, "Actual change and post-decision midpoint"),
        "14 text tokens": (1, 0, "Each token contains score and confidence"),
        "Event→Price MHA": (4, 3, "4 heads; price padding masked"),
        "Price→Event MHA": (5, 2, "4 heads; unavailable features masked"),
        "Fusion": (6, 3, "Residual price summary plus cross-attended context"),
        "Heads": (7, 3, "3-class direction and ordered q05/q50/q95"),
    }
    edges = [
        ("Yield panel", "2D CNN"),
        ("2D CNN", "Price Transformer"),
        ("Price Transformer", "Price tokens"),
        ("Price tokens", "Event→Price MHA"),
        ("Rate token", "Event→Price MHA"),
        ("14 text tokens", "Event→Price MHA"),
        ("Event→Price MHA", "Price→Event MHA"),
        ("Price Transformer", "Price→Event MHA"),
        ("Price→Event MHA", "Fusion"),
        ("Price Transformer", "Fusion"),
        ("Fusion", "Heads"),
    ]
    edge_x, edge_y = [], []
    for source, target in edges:
        x0, y0, _ = nodes[source]
        x1, y1, _ = nodes[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 2, "color": "#7f8c8d"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[value[0] for value in nodes.values()],
            y=[value[1] for value in nodes.values()],
            text=list(nodes),
            customdata=[value[2] for value in nodes.values()],
            mode="markers+text",
            textposition="bottom center",
            marker={"size": 34, "color": "#0072B2", "line": {"width": 2, "color": "white"}},
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Leakage-audited US10Y–FOMC cross-attention architecture",
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
        template="plotly_white",
        height=700,
    )
    output = PROJECT_ROOT / "docs" / "architecture_interactive.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True)
    print(output)
