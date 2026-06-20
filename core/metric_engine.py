def infer_kpis(df, semantics):
    kpis = []

    numeric  = semantics.get("numeric", [])
    id_cols  = semantics.get("id", [])
    cat_cols = semantics.get("categorical", [])

    # 1. Total Records — always first
    kpis.append(("Total Records", len(df)))

    # 2. Monetary / volume metrics — most useful KPIs
    for col in numeric:
        lname = col.lower().replace(" ", "").replace("_", "")
        # Skip binary/flag columns and percentage-like columns with tiny range
        if df[col].nunique() <= 5:
            continue
        # Totals for revenue/sales/profit/quantity/amount cols
        kpis.append((f"Total {col}", df[col].sum()))
        kpis.append((f"Avg {col}", df[col].mean()))
        if len(kpis) >= 9:
            break

    # 3. Unique count for ONE meaningful ID only (e.g. Order ID → transaction count)
    # Pick the most meaningful ID — prefer order/transaction over row/customer
    priority_keywords = ["order", "transaction", "invoice", "ticket", "sale"]
    best_id = None
    for kw in priority_keywords:
        for col in id_cols:
            if kw in col.lower():
                best_id = col
                break
        if best_id:
            break

    if best_id and len(kpis) < 8:
        kpis.append((f"Unique {best_id}", df[best_id].nunique()))

    # 4. Unique count for meaningful categoricals (e.g. # of Regions, Categories)
    useful_cats = [
        c for c in cat_cols
        if any(kw in c.lower() for kw in
               ["region", "category", "segment", "country", "state", "city",
                "product", "type", "status", "channel", "brand"])
        and df[c].nunique() <= 50
    ]
    for col in useful_cats[:2]:
        if len(kpis) >= 8:
            break
        kpis.append((f"# {col}s", df[col].nunique()))

    return kpis[:6]   # max 6 cards on dashboard
