from .llm_client import ask_llm


def generate_ai_insights(profile: dict, semantics: dict, patterns: dict, kpis: list) -> str:
    kpi_lines = "\n".join(
        f"  • {label}: {value:,.2f}" if isinstance(value, (int, float)) else f"  • {label}: {value}"
        for label, value in kpis[:6]
    )

    prompt = f"""You are a senior business data analyst reviewing a dataset dashboard.

DATASET STATS
  Rows: {profile.get('rows', '?')}  |  Columns: {profile.get('columns', '?')}
  Numeric: {semantics.get('numeric', [])[:4]}
  Categorical: {semantics.get('categorical', [])[:4]}
  Dates: {semantics.get('date', [])[:2]}

KEY METRICS
{kpi_lines}

PATTERNS
  Dominant metric (highest variance): {patterns.get('dominant_metric', 'N/A')}
  Business measures: {patterns.get('measures', [])[:5]}

INSTRUCTIONS
Write exactly 4 concise bullet-point insights (no headers, no preamble).
Each bullet must be actionable, data-specific, and business-focused.
Avoid restating the numbers verbatim — interpret their meaning.
Format: start each line with •
"""

    return ask_llm(prompt, max_tokens=500)
