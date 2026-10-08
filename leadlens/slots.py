"""Rule-based slot extraction -> completeness score -> clarifying question (English/Hinglish).
Rules (not an LLM) so behaviour is deterministic, auditable and free to serve."""
import re
from .data import CITIES, URGENT_WORDS

HINGLISH = ("chahiye", "batao", "bhejo", "rate", "karo", "hai", "mujhe", "hum", "me ", "ka ", "bhejiye", "karni")
SLOTS = ["quantity", "city", "timeline", "spec_or_use", "budget"]
WEIGHT = {"quantity": 0.30, "city": 0.25, "timeline": 0.20, "spec_or_use": 0.15, "budget": 0.10}
ASK_EN = {"quantity": "how many units / what quantity you need", "city": "your delivery city",
          "timeline": "when you need it", "spec_or_use": "the specification or intended use",
          "budget": "your approximate budget"}
ASK_HI = {"quantity": "kitni quantity chahiye", "city": "delivery kis shehar me chahiye",
          "timeline": "kab tak chahiye", "spec_or_use": "specification ya use kya hai",
          "budget": "approx budget kitna hai"}


def is_hinglish(text: str) -> bool:
    t = text.lower()
    return sum(w in t for w in HINGLISH) >= 2


def extract(text: str) -> dict:
    t = text.lower()
    return {
        "quantity": bool(re.search(r"\b\d[\d,]*\s*(pcs|units|kg|tons?|meters?|pieces?|nos)\b", t)),
        "city": any(c.lower() in t for c, _ in CITIES),
        "timeline": any(w in t for w in URGENT_WORDS + ("next month", "15 din", "15 days", "agle mahine")),
        "spec_or_use": bool(re.search(r"\b(for|specs?|grade|mm|inch|gsm|kw|factory|project|tender|resale|export|showroom)\b", t)),
        "budget": bool(re.search(r"(rs\.?|₹|budget)\s*\d", t)),
    }


def completeness(slots: dict) -> float:
    return round(sum(WEIGHT[k] for k, v in slots.items() if v), 2)


def clarifying_question(text: str, max_asks=2) -> str:
    slots = extract(text)
    missing = [k for k in SLOTS if not slots[k]][:max_asks]
    if not missing:
        return ""
    hi = is_hinglish(text)
    ask = ASK_HI if hi else ASK_EN
    parts = [ask[m] for m in missing]
    if hi:
        return "Best quotes ke liye please bataiye: " + " aur ".join(parts) + "."
    return "To get you the best quotes, could you tell us " + " and ".join(parts) + "?"
