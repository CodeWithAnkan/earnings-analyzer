import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHAVANTAGE_KEY")
BASE_URL = "https://www.alphavantage.co/query"

# All quarters to fetch per company
QUARTERS = [
    "2025Q3", "2025Q2", "2025Q1",
    "2024Q4", "2024Q3", "2024Q2", "2024Q1",
    "2023Q4"
]

def fetch_transcript(symbol: str, quarter: str) -> dict | None:
    """Fetch a single earnings call transcript from AlphaVantage."""
    res = requests.get(BASE_URL, params={
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": symbol.upper(),
        "quarter": quarter,
        "apikey": API_KEY
    }, timeout=15)

    if res.status_code != 200:
        return None

    data = res.json()

    # AlphaVantage returns empty transcript list if not found
    if not data.get("transcript"):
        return None

    return {
        "symbol": data["symbol"],
        "quarter": data["quarter"],
        "segments": data["transcript"]  # list of {speaker, title, content, sentiment}
    }

def fetch_all_transcripts(symbol: str, quarters: list[str] = None) -> list[dict]:
    """Fetch multiple quarters of transcripts for a symbol."""
    if quarters is None:
        quarters = QUARTERS

    results = []
    for quarter in quarters:
        data = fetch_transcript(symbol, quarter)
        if data and data["segments"]:
            print(f"  ✓ {quarter} — {len(data['segments'])} segments")
            results.append(data)
        else:
            print(f"  ✗ {quarter} — not found")

    return results