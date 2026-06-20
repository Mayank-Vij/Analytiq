import pandas as pd

def discover_patterns(df, profile):
    numeric_cols = profile["numeric_cols"]
    categorical_cols = profile["categorical_cols"]
    date_cols = profile["date_cols"]

    # Business metrics only (exclude IDs)
    measures = [
        c for c in numeric_cols
        if not c.lower().endswith("id")
        and not c.lower().startswith("row")
    ]

    dominant_metric = None
    if measures:
        dominant_metric = (
            df[measures]
            .select_dtypes(include="number")
            .var()
            .idxmax()
        )

    return {
        "measures": measures,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
        "dominant_metric": dominant_metric
    }
