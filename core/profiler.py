import pandas as pd

def profile_dataset(df):
    df = df.copy()

    # Step 1: detect date columns FIRST
    date_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.7:
                df[col] = parsed
                date_cols.append(col)

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)

    # Step 2: numeric columns (EXCLUDING datetimes)
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col not in date_cols
    ]

    # Step 3: categorical columns
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and col not in date_cols
    ]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols
    }
