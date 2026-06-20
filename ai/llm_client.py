import os

try:
    from groq import Groq
    _client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    _backend = "groq"
except Exception:
    _client = None
    _backend = "none"


def ask_llm(prompt: str, max_tokens: int = 600) -> str:
    if _backend == "groq" and _client:
        try:
            resp = _client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"⚠️ LLM unavailable: {e}"
    return "⚠️ No LLM backend configured. Set GROQ_API_KEY to enable AI insights."
