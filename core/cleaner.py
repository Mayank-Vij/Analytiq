import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Date detection FIRST (before numeric coercion)
    for col in df.columns:
        if df[col].dtype == "object" and any(
            kw in col.lower() for kw in ["date", "time", "year", "month", "day"]
        ):
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.6:
                df[col] = parsed
                continue

    # Numeric coercion for remaining object cols
    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = df[col].str.replace(",", "", regex=False).str.strip()
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().mean() > 0.7:
                df[col] = converted

    # Fill missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            pass
        else:
            df[col] = df[col].fillna("Unknown")

    return df
