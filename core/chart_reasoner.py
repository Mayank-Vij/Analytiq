"""
Decides which charts to render based on column semantics and profile.
Returns a list of chart specs consumed by the rendering layer.
"""


def select_charts(profile: dict, semantics: dict) -> list[dict]:
    specs = []

    date_cols = semantics.get("date", [])
    numeric_cols = semantics.get("numeric", [])
    categorical_cols = semantics.get("categorical", [])

    # Safe metrics = numeric cols that are not IDs
    safe_metrics = [
        c for c in numeric_cols
        if c in profile.get("numeric_cols", [])
        and not c.lower().endswith("id")
        and not c.lower().startswith("row")
    ]

    # ── Time series (if date + metric available) ─────────────────────────────
    if date_cols and safe_metrics:
        specs.append({
            "type": "time_series",
            "date": date_cols[0],
            "metric": safe_metrics[0],
            "label": f"{safe_metrics[0]} over time",
        })

    # ── Category bar (top contributor) ───────────────────────────────────────
    if categorical_cols and safe_metrics:
        specs.append({
            "type": "category_bar",
            "category": categorical_cols[0],
            "metric": safe_metrics[0],
            "label": f"{safe_metrics[0]} by {categorical_cols[0]}",
        })

    # ── Donut / pie (2nd categorical if available) ───────────────────────────
    if categorical_cols and safe_metrics:
        cat = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
        metric = safe_metrics[1] if len(safe_metrics) > 1 else safe_metrics[0]
        specs.append({
            "type": "pie",
            "category": cat,
            "metric": metric,
            "label": f"{metric} share by {cat}",
        })

    # ── Distribution histogram ────────────────────────────────────────────────
    if safe_metrics:
        specs.append({
            "type": "histogram",
            "column": safe_metrics[0],
            "label": f"Distribution of {safe_metrics[0]}",
        })

    # ── Correlation heatmap (when enough numeric cols) ────────────────────────
    if len(safe_metrics) >= 3:
        specs.append({
            "type": "correlation",
            "columns": safe_metrics,
            "label": "Correlation Matrix",
        })

    # ── Box plot (spread by category) ────────────────────────────────────────
    if categorical_cols and safe_metrics:
        specs.append({
            "type": "box",
            "category": categorical_cols[0],
            "metric": safe_metrics[0],
            "label": f"{safe_metrics[0]} spread by {categorical_cols[0]}",
        })

    # ── Scatter (two numeric cols) ────────────────────────────────────────────
    if len(safe_metrics) >= 2:
        specs.append({
            "type": "scatter",
            "x": safe_metrics[0],
            "y": safe_metrics[1],
            "label": f"{safe_metrics[0]} vs {safe_metrics[1]}",
        })

    return specs
