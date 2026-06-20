import pandas as pd

_ID_KEYWORDS = [
    "id", "code", "key", "number", "zip", "postal",
    "phone", "fax", "sku", "ref", "uuid", "num",
    "name", "email", "address", "url", "ip",
]

_MAX_RATIO = 0.05   # >5% unique values = ID-like
_MAX_UNIQUE = 50    # >50 unique values = ID-like


def classify_columns(df: pd.DataFrame) -> dict:
    semantics = {"numeric": [], "categorical": [], "date": [], "id": []}
    n = len(df)

    for col in df.columns:
        lname = col.lower().replace(" ", "").replace("_", "")

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            semantics["date"].append(col)
            continue

        if any(kw in lname for kw in ["date", "time", "year", "month", "day"]):
            semantics["date"].append(col)
            continue

        nunique = df[col].nunique()
        ratio = nunique / n if n > 0 else 0
        is_id_name = any(kw in lname for kw in _ID_KEYWORDS)

        if df[col].dtype == object:
            # ID by name OR too many unique values
            if is_id_name or ratio > _MAX_RATIO or nunique > _MAX_UNIQUE:
                semantics["id"].append(col)
            else:
                semantics["categorical"].append(col)
            continue

        if pd.api.types.is_integer_dtype(df[col]):
            if is_id_name or ratio > 0.9:
                semantics["id"].append(col)
                continue

        if df[col].dtype.kind in "if":
            semantics["numeric"].append(col)
            continue

        semantics["categorical"].append(col)

    return semantics