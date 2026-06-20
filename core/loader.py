import pandas as pd


def load_dataset(file):
    name = file.name

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)

    # CSV — try common encodings in order
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]
    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue

    # Last resort — ignore bad bytes
    file.seek(0)
    return pd.read_csv(file, encoding="utf-8", encoding_errors="replace")