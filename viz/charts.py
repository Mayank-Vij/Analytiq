import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Shared theme ───────────────────────────────────────────────────────────────
THEME = {
    "bg":      "#0F1117",
    "surface": "#1A1D27",
    "border":  "#2A2D3A",
    "accent":  "#6366F1",
    "accent2": "#22D3EE",
    "text":    "#E2E8F0",
    "muted":   "#64748B",
    "font":    "Inter, sans-serif",
}

_layout_defaults = dict(
    paper_bgcolor=THEME["surface"],
    plot_bgcolor=THEME["surface"],
    font=dict(family=THEME["font"], color=THEME["text"], size=13),
    title_font=dict(family=THEME["font"], size=15, color=THEME["text"]),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(
        gridcolor=THEME["border"], linecolor=THEME["border"],
        tickfont=dict(color=THEME["muted"]), title_font=dict(color=THEME["muted"]),
    ),
    yaxis=dict(
        gridcolor=THEME["border"], linecolor=THEME["border"],
        tickfont=dict(color=THEME["muted"]), title_font=dict(color=THEME["muted"]),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=THEME["muted"])),
    hoverlabel=dict(
        bgcolor=THEME["bg"], bordercolor=THEME["accent"],
        font=dict(family=THEME["font"], color=THEME["text"]),
    ),
)

# Colour palette for categorical charts
_PALETTE = [
    "#6366F1", "#22D3EE", "#F59E0B", "#10B981",
    "#EC4899", "#8B5CF6", "#F97316", "#14B8A6",
    "#EF4444", "#84CC16",
]
_OTHERS_COLOR = "#475569"   # slate for the "Others" bucket


# ── Helper: Top-N + Others aggregation ────────────────────────────────────────

def _topn_with_others(df, category_col, metric, top_n=10):
    """
    Returns a DataFrame with top_n rows by metric sum,
    plus an 'Others' row that sums everything else.
    If total distinct values <= top_n, returns all rows (no Others needed).
    """
    grouped = (
        df.groupby(category_col)[metric]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    if len(grouped) <= top_n:
        return grouped, False   # no Others needed

    top    = grouped.head(top_n).copy()
    others = grouped.iloc[top_n:]
    others_sum = others[metric].sum()
    others_row = pd.DataFrame({category_col: ["Others"], metric: [others_sum]})
    result = pd.concat([top, others_row], ignore_index=True)
    return result, True


# ── Chart builders ─────────────────────────────────────────────────────────────

def time_series(df, date_col, metric):
    if date_col not in df.columns or metric not in df.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return None
    if not pd.api.types.is_numeric_dtype(df[metric]):
        return None

    data = df.groupby(date_col, as_index=False)[metric].sum().sort_values(date_col)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[date_col], y=data[metric],
        mode="lines", fill="tozeroy",
        line=dict(color=THEME["accent"], width=2.5),
        fillcolor="rgba(99,102,241,0.12)",
        name=metric,
        hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:,.2f}}<extra></extra>",
    ))
    fig.update_layout(title=f"{metric} over time", **_layout_defaults)
    return fig


def category_bar(df, category_col, metric, top_n=10):
    if category_col not in df.columns or metric not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[metric]):
        return None

    data, has_others = _topn_with_others(df, category_col, metric, top_n)

    # Colour: accent for top bars, muted slate for Others
    colors = []
    for i, row in data.iterrows():
        if row[category_col] == "Others":
            colors.append(_OTHERS_COLOR)
        else:
            colors.append(_PALETTE[i % len(_PALETTE)])

    title = f"Top {top_n} {category_col} by {metric}" + (" + Others" if has_others else "")

    fig = go.Figure(go.Bar(
        x=data[category_col], y=data[metric],
        marker=dict(color=colors, line=dict(width=0), opacity=0.92),
        hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:,.2f}}<extra></extra>",
    ))
    fig.update_layout(title=title, **_layout_defaults)
    return fig


def histogram(df, column):
    if column not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    fig = go.Figure(go.Histogram(
        x=df[column], nbinsx=30,
        marker=dict(color=THEME["accent"], line=dict(color=THEME["border"], width=1), opacity=0.85),
        hovertemplate=f"{column}: %{{x}}<br>Count: %{{y}}<extra></extra>",
    ))
    fig.update_layout(title=f"Distribution of {column}", **_layout_defaults)
    return fig


def scatter_plot(df, x_col, y_col, color_col=None):
    if x_col not in df.columns or y_col not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        return None

    fig = go.Figure(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode="markers",
        marker=dict(color=THEME["accent"], size=7,
                    line=dict(color=THEME["border"], width=0.5), opacity=0.7),
        hovertemplate=f"{x_col}: %{{x:,.2f}}<br>{y_col}: %{{y:,.2f}}<extra></extra>",
    ))
    fig.update_layout(title=f"{x_col} vs {y_col}", **_layout_defaults)
    return fig


def correlation_heatmap(df, numeric_cols):
    if len(numeric_cols) < 2:
        return None

    cols = [c for c in numeric_cols if c in df.columns][:10]
    corr = df[cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0.0, "#0EA5E9"], [0.5, THEME["surface"]], [1.0, THEME["accent"]]],
        zmid=0,
        hoverongaps=False,
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=THEME["muted"]), outlinecolor=THEME["border"]),
    ))
    fig.update_layout(
        title="Correlation Matrix",
        xaxis=dict(tickfont=dict(color=THEME["muted"]), tickangle=-30),
        yaxis=dict(tickfont=dict(color=THEME["muted"])),
        **{k: v for k, v in _layout_defaults.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def pie_chart(df, category_col, metric, top_n=10):
    if category_col not in df.columns or metric not in df.columns:
        return None

    data, has_others = _topn_with_others(df, category_col, metric, top_n)

    # If Others slice > 60% of total, this chart is meaningless — skip it
    if has_others:
        total = data[metric].sum()
        others_val = data[data[category_col] == "Others"][metric].values
        if len(others_val) > 0 and total > 0:
            if others_val[0] / total > 0.60:
                return None

    # Build colour list — Others always gets the muted slate colour
    colors = []
    for _, row in data.iterrows():
        if row[category_col] == "Others":
            colors.append(_OTHERS_COLOR)
        else:
            colors.append(_PALETTE[len(colors) % len(_PALETTE)])

    title = (
        f"Top {top_n} {category_col} by {metric} (+ Others)"
        if has_others
        else f"{metric} by {category_col}"
    )

    fig = go.Figure(go.Pie(
        labels=data[category_col],
        values=data[metric],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=THEME["bg"], width=2)),
        textfont=dict(color=THEME["text"]),
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
        sort=False,   # keep our sort order (top → Others at end)
    ))
    fig.update_layout(
        title=title,
        **{k: v for k, v in _layout_defaults.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def box_plot(df, category_col, metric, top_n=10):
    if category_col not in df.columns or metric not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[metric]):
        return None

    top_cats = (
        df.groupby(category_col)[metric].sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    total_cats = df[category_col].nunique()
    has_others = total_cats > top_n

    # Top-N boxes
    fig = go.Figure()
    sub = df[df[category_col].isin(top_cats)]
    for i, cat in enumerate(top_cats):
        cat_data = sub[sub[category_col] == cat][metric]
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Box(
            y=cat_data, name=str(cat),
            marker_color=color, line_color=color,
            boxmean=True,
            hovertemplate=f"<b>{cat}</b><br>%{{y:,.2f}}<extra></extra>",
        ))

    # Others box — pool all remaining rows together
    if has_others:
        others_data = df[~df[category_col].isin(top_cats)][metric]
        fig.add_trace(go.Box(
            y=others_data, name="Others",
            marker_color=_OTHERS_COLOR, line_color=_OTHERS_COLOR,
            boxmean=True,
            hovertemplate="<b>Others</b><br>%{y:,.2f}<extra></extra>",
        ))

    title = (
        f"Top {top_n} {category_col} — {metric} distribution (+ Others)"
        if has_others
        else f"{metric} distribution by {category_col}"
    )
    fig.update_layout(title=title, **_layout_defaults)
    return fig